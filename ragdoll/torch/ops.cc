#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <THC/THC.h>

#include <cstdio>

#include "glog/logging.h"

#include "common/common.h"
#include "common/operations.h"

namespace ragdoll {

extern THCState *state;

class TorchTensor : public Tensor {
 public:
  TorchTensor(torch::Tensor tensor) : tensor_(tensor) {}

  void *GetData() override { return tensor_.data_ptr(); }
  gccl::gcclDataType_t GetGCCLDataType() override {
    switch (tensor_.scalar_type()) {
      case torch::kInt:
        return gccl::gcclDataType_t::gcclInt;
      case torch::kFloat:
        return gccl::gcclDataType_t::gcclFloat;
      default:
        CHECK(false) << "Unsupported data type " << tensor_.scalar_type();
    }
  }

 private:
  torch::Tensor tensor_;
};

torch::Tensor add_one_op(torch::Tensor z) { return torch::add(z, 1); }
void hello_op() { printf("Hello\n"); }
int rank_op() { return ragdoll_rank(); }

torch::Tensor graph_allgather_op(torch::Tensor t, int feat_size) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  ExecuteGraphAllgather(std::make_shared<TorchTensor>(t), feat_size, stream);
  return t;
}

torch::Tensor graph_allgather_backward_op(torch::Tensor t, int feat_size) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  ExecuteGraphAllgatherBackward(std::make_shared<TorchTensor>(t), feat_size, stream);
  return t;
}

PYBIND11_MODULE(ragdoll_torch_ops, m) {
  m.doc() = "Pytorch ops";
  m.def("add_one_op", &add_one_op, "Add one op");
  m.def("hello_op", &hello_op, "Hello op");
  m.def("rank_op", &rank_op, "Rank op");
  m.def("graph_allgather_op", &graph_allgather_op, "Graph allgather");
  m.def("graph_allgather_backward_op", &graph_allgather_backward_op,
        "Graph allgather backward");
}

}  // namespace ragdoll
