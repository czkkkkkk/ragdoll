#pragma once

#include "gccl.h"

namespace ragdoll {

class Tensor {
public:
  virtual ~Tensor() {}

  virtual void *GetData() = 0;
  virtual gccl::gcclDataType_t GetGCCLDataType() = 0;
};

} // namespace gccl