"""Torch Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
import dgl.function as fn
from . import GraphAllgatherFunc, GraphAllgatherOutplaceFunc


class GINConv(nn.Module):
    r"""Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    """

    def __init__(self,
                 apply_func,
                 aggregator_type,
                 n_nodes,
                 local_n_nodes,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))
        self._n_nodes = n_nodes
        self._local_n_nodes = local_n_nodes

    def pad_remote(self, t):
        padding = [0] * (2 * len(t.shape) - 2)
        padding += [0, self._n_nodes - self._local_n_nodes]
        ret = F.pad(t, padding)
        return ret

    def apply_allgather(self, t):
        t = GraphAllgatherOutplaceFunc.apply(
            t, self._n_nodes, self._local_n_nodes)
        # t = self.pad_remote(t)
        return t


    def slice_local(self, t):
        assert t.shape[0] == self._n_nodes
        t = t[:self._local_n_nodes, ...]
        return t

    def forward(self, graph, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        assert feat.shape[0] == self._local_n_nodes
        with graph.local_scope():
            feat_src = self.apply_allgather(feat)
            feat_dst = feat_src[:self._local_n_nodes]
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
            neigh = graph.dstdata['neigh']
            neigh = self.slice_local(neigh)
            rst = (1 + self.eps) * feat_dst + neigh
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
