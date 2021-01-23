"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from ragdoll.torch.graphconv import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 n_nodes,
                 local_n_nodes,
                 no_remote,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 comm_net):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, n_nodes, local_n_nodes, apply_gather=True, no_remote=True, norm='none', activation=activation, comm_net=comm_net))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, n_nodes, local_n_nodes, apply_gather=True, norm='none', activation=activation, comm_net=comm_net))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, n_nodes,
                                     local_n_nodes, apply_gather=True, no_remote=True, norm='none', comm_net=comm_net))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
