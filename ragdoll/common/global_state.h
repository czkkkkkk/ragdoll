#pragma once

#include "gccl.h"

namespace ragdoll {

struct GlobalState {
  bool initialized = false;
  int rank, n_peers;
  int device_id;
  gccl::gcclComm_t comm = nullptr;
  gccl::gcclCommInfo_t info = nullptr;
};

} // namespace ragdoll
