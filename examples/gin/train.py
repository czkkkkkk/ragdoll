"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import ragdoll
from ragdoll.data.datasets import Dataset
from ragdoll.data.syn_dataset import SynDataset

# sys
import numpy as np
import networkx as nx
import sys
import os
import atexit
import time

# torch
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.autograd.profiler as profiler

# self
from gin import GIN
from parser import Parser

# dgl
from dgl import DGLGraph
from dgl.data import register_data_args

def setup(rank, world_size, args):
    os.environ['RAGDOLL_USE_MPI'] = '0'
    os.environ['RAGDOLL_MASTER_ADDR'] = args.master_addr
    os.environ['RAGDOLL_PORT'] = '12308'
    os.environ['RAGDOLL_RANK'] = str(rank)
    os.environ['RAGDOLL_WORLD_SIZE'] = str(world_size)

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    if rank != 0:
        logs_file = args.logs_dir + '/r' + str(rank) + '.out'
        sys.stdout = open(logs_file, 'w')
    logs_name = 'GCCL.RANK.' + str(rank)
    ragdoll.init_logs(logs_name)

    init_method = 'tcp://{master_addr}:{master_port}'.format(
        master_addr=args.master_addr, master_port='12356')
    dist.init_process_group('nccl', init_method=init_method,
                            rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), correct.item() * 1.0, len(labels)


def run(rank, world_size, args):
    print('Running DDP on rank', rank)
    setup(rank, world_size, args)
    ragdoll.init()
    dev_id = ragdoll.device_id()
    if len(args.input_graph) > 0 or len(args.cached_dir) > 0:
        data = SynDataset(rank == 0, args)
    else:
        data = Dataset(rank == 0, args)

    feat_size = args.feat_size
    n_classes = args.n_classes

    torch.cuda.set_device(dev_id)
    features = torch.FloatTensor(data.features).cuda()
    labels = torch.LongTensor(data.labels).cuda()
    labels = torch.LongTensor([0]).cuda()
    train_mask = torch.BoolTensor(data.train_mask).cuda()
    val_mask = torch.BoolTensor(data.val_mask).cuda()
    test_mask = torch.BoolTensor(data.test_mask).cuda()

    n_classes = args.n_classes
    n_nodes = data.n_nodes
    local_n_nodes = data.local_n_nodes

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        feat_size, args.hidden_dim, n_classes,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type, n_nodes, local_n_nodes)

    model.cuda()
    model = DDP(model, device_ids=[dev_id])
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)
    optimizer.zero_grad()

    print("Start training")
    dur = []
    for epoch in range(args.epochs):
        model.train()
        torch.distributed.barrier()
        if epoch >= 3:
            t0 = time.time()
        # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        logits = model(data.graph, features)
        loss = loss_fcn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.current_stream().synchronize()
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        if epoch >= 3:
            dur.append(time.time() - t0)
        print('Peak memory is {} GB'.format(torch.cuda.max_memory_allocated(dev_id) / 1e9))

        print('acc is {}, loss is {}, this epoch using time {} s, avg time {} s.'.format(
            0, loss.item(), dur[-1] if epoch >= 3 else 0, np.mean(dur)))


    cleanup()


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def SetArgs(args):
    if len(args.input_graph) or len(args.cached_dir) > 0:
        return
    if args.dataset == 'cora':
        args.feat_size = 1433
        args.n_classes = 7
    elif args.dataset == 'citeseer':
        args.feat_size = 3703
        args.n_classes = 6
    elif args.dataset == 'pubmed':
        args.feat_size = 500
        args.n_classes = 3
    elif args.dataset == 'reddit':
        args.feat_size = 602
        args.n_classes = 41
    else:
        assert False


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')

    SetArgs(args)
    print(args)

    nproc = args.world_size
    world_size = args.world_size
    ranks = [i for i in range(nproc)]
    if args.nproc_per_node != -1:
        nproc = args.nproc_per_node
        node_rank = args.node_rank
        ranks = [i + node_rank * args.nproc_per_node for i in range(nproc)]
    print('ranks is ', ranks, ' nproc is', nproc)

    processes = []
    for i in range(nproc):
        p = mp.Process(target=run, args=(ranks[i], world_size, args))
        atexit.register(kill_proc, p)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
