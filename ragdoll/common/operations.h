#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "common/common.h"

namespace ragdoll {

extern "C" {

// explicity export the symbols since pybind will disable them

__attribute__((visibility("default"))) void ragdoll_hello();

__attribute__((visibility("default"))) void ragdoll_init();

__attribute__((visibility("default"))) void ragdoll_init_logs(const char *file);

__attribute__((visibility("default"))) int ragdoll_rank();
__attribute__((visibility("default"))) int ragdoll_device_id();
__attribute__((visibility("default"))) int ragdoll_world_size();

__attribute__((visibility("default"))) void ragdoll_partition_graph(
    int n_nodes, int *xadj, int *adjncy, int *sgn, int **sg_xadj,
    int **sg_adjncy);

__attribute__((visibility("default"))) void ragdoll_partition_graph_on_dir(
    const char *dirname, int *sgn, int **sg_xadj, int **sg_adjncy);

__attribute__((visibility("default"))) void ragdoll_release(int *ptr);

__attribute__((visibility("default"))) void ragdoll_dispatch_float(
    float *data, int feat_size, int local_n_nodes, float *local_data,
    int no_remote);

__attribute__((visibility("default"))) void ragdoll_dispatch_int(
    int *data, int feat_size, int local_n_nodes, int *local_data,
    int no_remote);

__attribute__((visibility("default"))) int ragdoll_get_local_n_nodes();

// void ragdoll_distribute_data();

// void ragdoll_graph_allgather();

// void ragdoll_graph_allreduce();
}

void ExecuteGraphAllgather(std::shared_ptr<Tensor> tensor, int feat_size,
                           cudaStream_t stream);
void ExecuteGraphAllgatherBackward(std::shared_ptr<Tensor> tensor,
                                   int feat_size, cudaStream_t stream);

}  // namespace ragdoll
