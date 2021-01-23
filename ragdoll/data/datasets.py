import ragdoll
from dgl.data import load_data
from ragdoll.data.data_wrapper import DataWrapper
import networkx as nx
import numpy as np
import scipy
from dgl import DGLGraph
import torch


class Dataset(object):
    def __init__(self, is_root, args):
        self.name = args.dataset
        self.is_root = is_root
        self._load(args)

    def pad_to_128(self, a):
        assert len(a.shape) == 2
        d = a.shape[1]
        padding = (128 - d % 128) % 128
        if padding == 0:
            return
        return np.pad(a, ((0, 0), (0, padding)), constant_values=0)

    def _load(self, args):
        data = None
        n_nodes = None
        if self.is_root:
            data = load_data(args)
            n_nodes = data.graph.number_of_nodes()
            if args.dataset == 'reddit':
                data.graph = data.graph.adjacency_matrix_scipy()
            else:
                data.graph = nx.to_scipy_sparse_matrix(data.graph)

            print('n classes:', data.num_labels)
            print('feat size', data.features.shape[-1])
        data = DataWrapper(data)
        n_nodes = DataWrapper(n_nodes)

        g = data.get_attr('graph')
        indptr = g.get_attr('indptr')
        indices = g.get_attr('indices')

        print('Try to partition')
        sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            n_nodes.get_val(0), indptr.get_val([]), indices.get_val([]))

        sg_e = sg_xadj[sg_n]
        assert sg_e == len(sg_adjncy)
        assert sg_n + 1 == len(sg_xadj)
        for u in sg_adjncy:
            assert u >= 0 and u < sg_n
        for i in range(sg_n):
            assert sg_xadj[i + 1] >= sg_xadj[i]

        edge_data = np.ones([sg_e])
        print('Building csr matrix')
        subgraph = scipy.sparse.csr_matrix(
            (edge_data, sg_adjncy, sg_xadj), shape=[sg_n, sg_n])
        print('Build csr matrix done')

        self.local_n_nodes = ragdoll.get_local_n_nodes()
        self.n_nodes = sg_n

        #print('To nx sparse matrix')

        #self.graph = nx.from_scipy_sparse_matrix(
        #    subgraph, create_using=nx.DiGraph())
        #print('To nx sparse matrix Done')
        self.graph = subgraph

        features = data.get_attr('features')
        labels = data.get_attr('labels').call_func('astype', np.int32)
        train_mask = data.get_attr('train_mask').call_func('astype', np.int32)
        val_mask = data.get_attr('val_mask').call_func('astype', np.int32)
        test_mask = data.get_attr('test_mask').call_func('astype', np.int32)

        if self.is_root:
            print('feature shape is', features.get_val().shape)
            print('labels shape is', labels.get_val().shape)
            print('train mask shape is', train_mask.get_val().shape)

        features = ragdoll.dispatch_float(
            features.get_val(), args.feat_size, sg_n, no_remote=1)[:self.local_n_nodes*args.feat_size]
        self.features = np.reshape(features, [-1, args.feat_size])
        self.features = self.pad_to_128(self.features)
        args.feat_size = self.features.shape[-1]
        labels = ragdoll.dispatch_int(labels.get_val(), 1, sg_n, no_remote=1)[
            :self.local_n_nodes]
        train_mask = ragdoll.dispatch_int(
            train_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        val_mask = ragdoll.dispatch_int(
            val_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        test_mask = ragdoll.dispatch_int(
            test_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        self.labels = np.reshape(labels, [-1])
        self.train_mask = np.reshape(train_mask, [-1])
        self.val_mask = np.reshape(val_mask, [-1])
        self.test_mask = np.reshape(test_mask, [-1])

        print('My feature shape is', self.features.shape)
        print('My labels shape is', self.labels.shape)
        print('My train mask shape is', self.train_mask.shape)
        print('My val mask shape is', self.val_mask.shape)
        print('My test mask shape is', self.test_mask.shape)

        g = self.graph

        #if hasattr(args, 'self_loop'):
        #    if args.self_loop:
        #        g.remove_edges_from(nx.selfloop_edges(g))
        #        g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
        self.graph = g
