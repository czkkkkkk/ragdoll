"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse

from dgl.data import register_data_args


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        register_data_args(self.parser)
        self.parser.add_argument(
            '--batch_size', type=int, default=32,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--filename', type=str, default="",
            help='output file')

        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument(
            '--num_layers', type=int, default=2,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=512,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--epochs', type=int, default=20,
            help='number of epochs to train (default: 350)')
        self.parser.add_argument(
            '--lr', type=float, default=0.01,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.5,
            help='final layer dropout (default: 0.5)')
        self.parser.add_argument("--self-loop", action='store_true',
                            help="graph self-loop (default=False)")
        self.parser.add_argument("--feat_size", type=int,
                            default=256, help="feature size")
        self.parser.add_argument("--n_classes", type=int,
                            default=1, help="world size")

        self.parser.add_argument("--world_size", type=int,
                            default=3, help="world size")

        self.parser.add_argument("--logs_dir", type=str,
                            default="./logs", help="logs dir")

        self.parser.add_argument("--input_graph", type=str,
                            default="", help="input graph")
        self.parser.add_argument("--cached_dir", type=str,
                            default="", help="Cached dir")
        self.parser.add_argument("--local_rank", default=-1, type=int, help="Distributed local rank")
        self.parser.add_argument("--node_rank", default=-1, type=int, help="Distributed node_rank")
        self.parser.add_argument("--nproc_per_node", default=-1, type=int, help="Distributed process per node")
        self.parser.add_argument("--master_addr", default="localhost", type=str, help="Master address")

        # done
        self.args = self.parser.parse_args()
