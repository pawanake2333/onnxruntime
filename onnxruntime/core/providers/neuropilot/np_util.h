#pragma once

#include <cstdio>
#include <string>
#include <type_traits>
#include <dlfcn.h>

#include "core/graph/graph.h"


namespace onnxruntime {

class DymLoader {
 public:
  DymLoader(const std::string& name, int flag, bool freedl = false);
  ~DymLoader();

 protected:
  template <typename Sig>
  std::add_pointer_t<Sig> sym(const char* name);

 protected:
  void* handle_;
  std::string name_;
  bool free_;
};

template <typename Sig>
std::add_pointer_t<Sig> DymLoader::sym(const char* name) {
  return (std::add_pointer_t<Sig>)dlsym(handle_, name);
}

bool support_neuropilot();

struct ArgShapeInfo {
  std::vector<int64_t> shape;
  int64_t elem_count_all;
  int64_t elem_size_bits;
};

ArgShapeInfo get_arg_shape_size(const NodeArg& arg);

}  // namespace onnxruntime

#define LoadSym(method) method = sym<decltype(::method)>(#method)
#define DeclareSym(method) decltype(::method)* method
