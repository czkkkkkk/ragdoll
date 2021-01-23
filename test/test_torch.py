import unittest
import os

import ragdoll
import numpy as np
import torch


class TorchTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """
    @classmethod
    def setUpClass(cls):
        ragdoll.init()

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)

    def check_array_eq(self, lhs, rhs):
        assert len(lhs) == len(rhs)
        for l, r in zip(lhs, rhs):
            assert l == r

    def test_init(self):
        rank = ragdoll.rank_op()
        print('My rank is', rank)

    def test_partition_graph(self):
        rank = ragdoll.rank()
        world_size = ragdoll.world_size()
        if world_size != 3:
            return

        n_nodes = 7
        xadj = [0, 2, 4, 6, 6, 6, 6, 6]
        adjncy = [1, 2, 3, 4, 5, 6]
        os.environ['GCCL_PART_OPT'] = 'NAIVE'

        sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            n_nodes, xadj, adjncy)
        if rank == 0:
            assert sg_n == 5
            assert sg_xadj == [0, 0, 0, 0, 1, 2]
            assert sg_adjncy == [1, 2]
        else:
            assert sg_n == 3
            assert sg_xadj == [0, 1, 1, 2]
            assert sg_adjncy == [1, 0]

    def test_graph_allgather(self):
        rank = ragdoll.rank()
        world_size = ragdoll.world_size()
        if world_size != 3:
            return
        dev_id = rank

        n_nodes = 7
        xadj = [0, 2, 4, 6, 6, 6, 6, 6]
        adjncy = [1, 2, 3, 4, 5, 6]
        os.environ['GCCL_PART_OPT'] = 'NAIVE'

        sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            n_nodes, xadj, adjncy)
        feat_size = 128
        if rank == 0:
            cpu_input = [0, 3, 6, -1, -1]
            exp_output = [0, 3, 6, 1, 2]
        elif rank == 1:
            cpu_input = [1, 4, -1]
            exp_output = [1, 4, 0]
        else:
            cpu_input = [2, 5, -1]
            exp_output = [2, 5, 0]

        cpu_input = np.repeat(cpu_input, feat_size)
        exp_output = np.repeat(exp_output, feat_size)
        th_input = torch.Tensor(cpu_input).cuda(dev_id)
        th_output = ragdoll.graph_allgather(th_input, feat_size)
        th_output = th_output.cpu().detach().numpy()
        self.check_array_eq(exp_output, th_output)

    def test_dispatch_data(self):
        rank = ragdoll.rank()
        world_size = ragdoll.world_size()
        feat_size = 3
        if world_size != 3:
            return
        dev_id = rank

        n_nodes = 7
        xadj = [0, 2, 4, 6, 6, 6, 6, 6]
        adjncy = [1, 2, 3, 4, 5, 6]
        os.environ['GCCL_PART_OPT'] = 'NAIVE'

        sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            n_nodes, xadj, adjncy)
        data = None
        if rank == 0:
            data = np.array([7, 1, 2, 3, 4, 5, 6])
            exp_data = np.array([7, 3, 6, 1, 2])
        elif rank == 1:
            exp_data = np.array([1, 4, 7])
        else:
            exp_data = np.array([2, 5, 7])
        if data is not None:
            data = np.repeat(data, feat_size)
        exp_data = np.repeat(exp_data, feat_size)
        local_data = ragdoll.dispatch_float(data, feat_size, sg_n)
        self.check_array_eq(local_data, exp_data)


if __name__ == '__main__':
    unittest.main()
