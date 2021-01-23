"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

import dgl.function as fn
from . import GraphAllgatherFunc, GraphAllgatherOutplaceFunc

# pylint: disable=W0235


class GraphConv(nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 n_nodes,
                 local_n_nodes,
                 apply_gather=False,
                 no_remote=True,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 comm_net=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            assert False
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._n_nodes = n_nodes
        self._local_n_nodes = local_n_nodes
        self._no_remote = no_remote
        self._apply_gather = apply_gather
        self._comm_net = comm_net

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            assert False
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        if self._comm_net:
            self.comm_net_w = nn.Parameter(th.Tensor(in_feats, out_feats))

        self.reset_parameters_constant()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def reset_parameters_constant(self):
        if self.weight is not None:
            init.constant_(self.weight, 0.5)
        if self.bias is not None:
            init.zeros_(self.bias)

    def apply_allgather(self, t):
        if not self._apply_gather:
            return t
        if self._no_remote:
            t = GraphAllgatherOutplaceFunc.apply(
                t, self._n_nodes, self._local_n_nodes)
        else:
            t = GraphAllgatherFunc.apply(t)
        return t

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if self._norm == 'both':
                degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                feat = feat * norm

            weight = self.weight

            if self._in_feats > self._out_feats and False:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    assert feat.shape[-1] == weight.shape[-2], "feat shape is {}, weight shape is {}".format(
                        feat.shape, weight.shape)
                    feat = th.matmul(feat, weight)
                feat = self.apply_allgather(feat)
                graph.srcdata['h'] = feat
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                rst = rst[:self._local_n_nodes, :]
            else:
                # aggregate first then mult W
                feat = self.apply_allgather(feat)
                graph.srcdata['h'] = feat
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                rst = rst[:self._local_n_nodes, :]
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._comm_net:
                rst += th.matmul(feat[:self._local_n_nodes], self.comm_net_w)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
