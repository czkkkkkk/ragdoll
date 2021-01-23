#pragma once

#include <cstdlib>
#include <sstream>
#include <string>

namespace ragdoll {

template <typename T>
T GetEnvParam(const std::string &key, T default_value) {
  auto gccl_str = std::string("RAGDOLL_") + key;
  char *ptr = std::getenv(gccl_str.c_str());
  if (ptr == nullptr) return default_value;
  std::stringstream converter(ptr);
  T ret;
  converter >> ret;
  return ret;
}

template <typename T>
T GetEnvParam(const std::string &key) {
  auto gccl_str = std::string("RAGDOLL_") + key;
  char *ptr = std::getenv(gccl_str.c_str());
  CHECK(ptr != nullptr) << "Environment variable " << gccl_str
                        << " does not exist";
  std::stringstream converter(ptr);
  T ret;
  converter >> ret;
  return ret;
}

template <typename T>
T GetEnvParam(const char *str, T default_value) {
  return GetEnvParam<T>(std::string(str), default_value);
}
}  // namespace ragdoll
