import ragdoll
import sys
from dgl.data import register_data_args
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import atexit
import numpy as np
import torch.optim as optim
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from gcn import GCN
import torch.nn.functional as F
from ragdoll.data.datasets import Dataset
from ragdoll.data.syn_dataset import SynDataset


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
    ragdoll.init()
    ragdoll.set_comm_pattern(args.comm)

    dev_id = ragdoll.device_id()
    torch.cuda.set_device(dev_id)

    init_method = 'tcp://{master_addr}:{master_port}'.format(
        master_addr=args.master_addr, master_port='16389')
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
    print('Running DDP on rank', rank, 'world size', world_size)

    setup(rank, world_size, args)
    dev_id = ragdoll.device_id()

    if len(args.input_graph) > 0 or len(args.cached_dir) > 0:
        data = SynDataset(rank == 0, args)
    else:
        data = Dataset(rank == 0, args)

    feat_size = args.feat_size

    features = torch.FloatTensor(data.features).cuda()
    labels = torch.LongTensor(data.labels).cuda()
    train_mask = torch.BoolTensor(data.train_mask).cuda()
    val_mask = torch.BoolTensor(data.val_mask).cuda()
    test_mask = torch.BoolTensor(data.test_mask).cuda()

    n_classes = args.n_classes
    n_nodes = data.n_nodes
    local_n_nodes = data.local_n_nodes

    model = GCN(data.graph, n_nodes, local_n_nodes, True, feat_size, args.n_hidden, n_classes,
                args.n_layers, F.relu, args.dropout, comm_net=args.comm_net)
    model.cuda()
    model = DDP(model, device_ids=[dev_id])
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    optimizer.zero_grad()

    dur = []
    print("Start training... for {} epochs".format(args.n_epochs))
    for epoch in range(args.n_epochs):
        print('Epoch {} -------------'.format(epoch))
        model.train()
        torch.distributed.barrier()
        if epoch >= 3:
            t0 = time.time()

        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        if epoch >= 3:
            dur.append(time.time() - t0)
        # acc, _, _ = evaluate(model, features, labels, val_mask)
        # print('acc is {}, loss is {}, this epoch using time {}, avg time {}.'.format(
        #    acc, loss.item(), dur[-1] if epoch >= 3 else 0, np.mean(dur)))
        print('Using time to synchronize model', t2 - t1)
        print('Peak memory is {} GB'.format(
            torch.cuda.max_memory_allocated(dev_id) / 1e9))
        print('this epoch uses time {} s, avg time {} s.'.format(
            dur[-1] if epoch >= 3 else 0, np.mean(dur)))

    ##acc, corr, total = evaluate(model, features, labels, test_mask)
    ##print('my corr is', corr, 'my total is', total)
    ##corr = torch.Tensor([corr]).cuda(dev_id)
    ##total = torch.Tensor([total]).cuda(dev_id)
    ##corrs, totals = [], []
    ##for i in range(world_size):
    ##    corrs.append(torch.Tensor([0]).cuda(dev_id))
    ##    totals.append(torch.Tensor([0]).cuda(dev_id))
    ##torch.distributed.all_gather(corrs, corr)
    ##torch.distributed.all_gather(totals, total)
    ##print('corrs is', corrs)
    ##print('totals is', totals)
    ##corr = torch.stack(corrs, dim=0).sum(dim=0).item() * 1.0
    ##total = torch.stack(totals, dim=0).sum(dim=0).item() * 1.0
    ##print('Test acc is', corr / total)

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
    # citeseer feat size: 3703, n_classes 6
    # cora feat size: 1433, n_classes 7
    # pubmed feat size: 500, n_classes 3
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--feat_size", type=int,
                        default=256, help="feature size")
    parser.add_argument("--n_classes", type=int,
                        default=3, help="world size")
    parser.add_argument("--comm_net", type=bool,
                        default=False, help="Comm net")

    parser.add_argument("--input_graph", type=str,
                        default="", help="input graph")
    parser.add_argument("--world_size", type=int,
                        default=3, help="world size")

    parser.add_argument("--logs_dir", type=str,
                        default="./logs", help="logs dir")
    parser.add_argument("--cached_dir", type=str,
                        default="", help="Cached dir")

    parser.add_argument("--comm", type=str,
                        default="greedy", help="communication pattern")

    parser.add_argument("--local_rank", default=-1,
                        type=int, help="Distributed local rank")
    parser.add_argument("--node_rank", default=-1, type=int,
                        help="Distributed node_rank")
    parser.add_argument("--nproc_per_node", default=-1,
                        type=int, help="Distributed process per node")
    parser.add_argument("--master_addr", default="localhost",
                        type=str, help="Master address")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    SetArgs(args)
    print(args)
    # ragdoll.init_logs(args.logs_dir)
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
