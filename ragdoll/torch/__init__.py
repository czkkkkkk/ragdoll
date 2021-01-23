from .. import graph_allgather_op, graph_allgather_backward_op
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

use_comm = 'ag'

def set_comm_pattern(pattern):
    global use_comm
    if pattern == 'dgcl' or pattern == 'alltoall':
        use_comm = 'ag' 
    elif pattern == 'swap':
        use_comm = 'shm'
    elif pattern == 'nocomm' or pattern == 'replication':
        use_comm = 'no'
    else:
        assert False
        
class GraphAllgatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        feat_size = np.prod(t.shape[1:])
        assert feat_size % 4 == 0, "feat_size is {}".format(feat_size)
        ctx.mark_dirty(t)
        graph_allgather_op(t, feat_size)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        feat_size = np.prod(grad_output.shape[1:])
        graph_allgather_backward_op(grad_output, feat_size)
        return grad_output


class GraphAllgatherOutplaceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, n_nodes, local_n_nodes):
        ctx.local_n_nodes = local_n_nodes
        feat_size = np.prod(t.shape[1:])
        assert feat_size % 4 == 0, "feat_size is {}".format(feat_size)
        ret = t.new(t)
        if use_comm == 'shm':
            dist.barrier()
            dev_id = ret.get_device()
            if dev_id % 2 == 0:
                ret = ret.cpu()
                ret = ret.cuda(dev_id)
            dist.barrier()
        padding = [0] * (2 * len(t.shape) - 2)
        padding += [0, n_nodes - local_n_nodes]
        ret = F.pad(ret, padding)
        if use_comm == 'ag':
# print('Allgather of feat size', feat_size)
            graph_allgather_op(ret, feat_size)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        local_n_nodes = ctx.local_n_nodes
        feat_size = np.prod(grad_output.shape[1:])
        global use_comm
        if use_comm == 'ag':
# print('Allgather backward of feat size', feat_size)
            graph_allgather_backward_op(grad_output, feat_size)

        ret = grad_output[:local_n_nodes, ...]
        if use_comm == 'shm':
            dist.barrier()
            dev_id = ret.get_device()
            if dev_id % 2 == 0:
                ret = ret.cpu()
                ret = ret.cuda(dev_id)
            dist.barrier()
        return ret, None, None
