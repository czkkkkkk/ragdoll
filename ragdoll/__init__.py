
import os
import sys
import time

# Must set before torch
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"  # noqa
import torch.distributed as distributed

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)  # noqa
from . import ragdoll_torch_ops  # pylint: disable=import-error
from .ragdoll_core import RagdollCore

core = RagdollCore()

# ctypes
hello = core.hello
init = core.init
set_comm_pattern=core.set_comm_pattern
init_logs = core.init_logs
rank = core.rank
device_id = core.device_id
world_size = core.world_size
partition_graph = core.partition_graph
partition_graph_on_dir = core.partition_graph_on_dir
dispatch_float = core.dispatch_float
dispatch_int = core.dispatch_int
get_local_n_nodes = core.get_local_n_nodes


# ops
add_one_op = ragdoll_torch_ops.add_one_op
rank_op = ragdoll_torch_ops.rank_op
graph_allgather_op = ragdoll_torch_ops.graph_allgather_op
graph_allgather_backward_op = ragdoll_torch_ops.graph_allgather_backward_op
