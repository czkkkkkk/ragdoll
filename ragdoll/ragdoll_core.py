import ctypes
import pathlib
import sysconfig
import numpy as np



def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


class RagdollCore(object):
    def __init__(self):
        lib_name = "ragdoll_torch_ops" + get_ext_suffix()
        so_path = pathlib.Path(__file__).with_name(lib_name)
        # so_path = str(pathlib.Path(__file__).with_name('libragdoll_core.so'))
        print('so path is ', so_path)
        self.lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    def hello(self):
        self.lib.ragdoll_hello()

    def set_comm_pattern(self, comm):
        from . import torch
        torch.set_comm_pattern(comm)

    def init_logs(self, log_file):
        log_file = log_file.encode('utf-8')
        self.lib.ragdoll_init_logs(log_file)

    def init(self):
        self.lib.ragdoll_init()

    def rank(self):
        return self.lib.ragdoll_rank()

    def device_id(self):
        return self.lib.ragdoll_device_id()

    def world_size(self):
        return self.lib.ragdoll_world_size()

    def partition_graph(self, n_nodes, xadj, adjncy):
        c_xadj = (ctypes.c_int * len(xadj))(*xadj)
        c_adjncy = (ctypes.c_int * len(adjncy))(*adjncy)
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        self.lib.ragdoll_partition_graph(n_nodes, c_xadj, c_adjncy, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy))

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        py_sg_xadj = [sg_xadj[i] for i in range(sg_n.value + 1)]
        py_sg_adjncy = [sg_adjncy[i] for i in range(n_edges)]
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy

    def partition_graph_on_dir(self, dirname):
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        dirname = dirname.encode('utf-8')
        self.lib.ragdoll_partition_graph_on_dir(dirname, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy))

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        py_sg_xadj = [sg_xadj[i] for i in range(sg_n.value + 1)]
        py_sg_adjncy = [sg_adjncy[i] for i in range(n_edges)]
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy

    def dispatch_float(self, t, feat_size, local_n_nodes, no_remote=0):
        if t is None:
            t = np.array([])
        t = t.ravel().tolist()
        c_ptr = (ctypes.c_float * len(t))(*t)
        local_data = (ctypes.c_float * (local_n_nodes * feat_size))()
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        return [local_data[v] for v in range(local_n_nodes * feat_size)]

    def dispatch_int(self, t, feat_size, local_n_nodes, no_remote=0):
        if t is None:
            t = np.array([])
        t = t.ravel().tolist()
        c_ptr = (ctypes.c_int * len(t))(*t)
        local_data = (ctypes.c_int * (local_n_nodes * feat_size))()
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        return [local_data[v] for v in range(local_n_nodes * feat_size)]

    def get_local_n_nodes(self):
        return self.lib.ragdoll_get_local_n_nodes()
