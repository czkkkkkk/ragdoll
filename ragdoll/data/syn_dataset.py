import ragdoll
from ragdoll.data.data_wrapper import DataWrapper
from dgl import DGLGraph

import networkx as nx
import scipy
import numpy as np
import time


def read_csr(file):
    with open(file, "r") as f:
        n, m = map(int, f.readline().split())
        start = time.time()
        print('Loading csr graph with {} nodes and {} edges'.format(n, m))
        n_xadj = int(f.readline())
        xadj = f.readline().split()
        n_adjncy = int(f.readline())
        adjncy = f.readline().split()
        edge_data = [1] * len(adjncy)
        graph = scipy.sparse.csr_matrix(
            (edge_data, adjncy, xadj), shape=[n, n])
        end = time.time()
        print('Finished loading, using time ', end - start)

    return n, m, graph


def read_adj(file):
    G = nx.DiGraph()
    with open(file, "r") as f:
        n, m = map(int, f.readline().split())
        edges = []
        for i in range(m):
            u, v = map(int, f.readline().split())
            edges.append([u, v])
        G.add_edges_from(edges)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    G = nx.to_scipy_sparse_matrix(G)
    return n_nodes, n_edges, G


class SynDataset(object):
    def __init__(self, is_root, args):
        self.is_root = is_root
        self._load(args)

    def _load(self, args):
        graph, n_nodes = None, None
        if len(args.cached_dir) == 0:
            if self.is_root:
                print('Loading graph...')
                if '-csr' in args.input_graph:
                    n_nodes, n_edges, graph = read_csr(args.input_graph)
                else:
                    n_nodes, n_edges, graph = read_adj(args.input_graph)
                print('Number of nodes is {}, number of edges is {}'.format(
                    n_nodes, n_edges))
                print('Converting graph to sparse matrix')

            graph = DataWrapper(graph)
            n_nodes = DataWrapper(n_nodes)

            indptr = graph.get_attr('indptr')
            indices = graph.get_attr('indices')

            print('Try to partition')
            sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
                n_nodes.get_val(0), indptr.get_val([]), indices.get_val([]))
        else:
            sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph_on_dir(
                args.cached_dir)

        sg_e = sg_xadj[sg_n]
        assert sg_e == len(sg_adjncy)
        assert sg_n + 1 == len(sg_xadj)
        for u in sg_adjncy:
            assert u >= 0 and u < sg_n
        for i in range(sg_n):
            assert sg_xadj[i + 1] >= sg_xadj[i]

        edge_data = np.ones([sg_e])
        start = time.time()
        subgraph = scipy.sparse.csr_matrix(
            (edge_data, sg_adjncy, sg_xadj), shape=[sg_n, sg_n])
        end = time.time()
        print('Using time to build csr matrix', end - start)

        self.local_n_nodes = ragdoll.get_local_n_nodes()
        self.n_nodes = sg_n

        self.graph = subgraph
        feat_size = args.feat_size
        self.features = np.ones([self.local_n_nodes, feat_size])
        self.labels = np.ones([self.local_n_nodes])
        self.train_mask = np.ones([self.local_n_nodes])
        self.val_mask = np.ones([self.local_n_nodes])
        self.test_mask = np.ones([self.local_n_nodes])
        self.graph = DGLGraph(self.graph)
