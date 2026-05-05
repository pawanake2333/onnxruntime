
#include "np_util.h"

#include <cstdio>
#include <mutex>
#include <dlfcn.h>

#include "core/common/common.h"

namespace onnxruntime {

static const char* kNeuropilotCheckValidPath = "libapuwareutils_v2.mtk.so";

DymLoader::DymLoader(const std::string& name, int flag, bool freedl) : handle_(nullptr), name_(name), free_(freedl) {
  handle_ = dlopen(name_.c_str(), flag);
  if (handle_ == nullptr) {
    ORT_THROW("Failed to dlopen ", name_, ". Error: ", dlerror());
  }
}

DymLoader::~DymLoader() {
  if (handle_ && free_) {
    dlclose(handle_);
  }
}

bool support_neuropilot() {
  static std::once_flag flag;
  static bool supported = false;
  std::call_once(flag, [&]() {
    void* handle = dlopen(kNeuropilotCheckValidPath, RTLD_LAZY);
    supported = handle != nullptr;
  });
  return supported;
}

ArgShapeInfo get_arg_shape_size(const NodeArg& arg) {
  ArgShapeInfo info;
  auto elem_type = arg.TypeAsProto()->tensor_type().elem_type();
  switch (elem_type) {
    case (int)onnx::TensorProto_DataType_UINT8:
    case (int)onnx::TensorProto_DataType_INT8:
    case (int)onnx::TensorProto_DataType_BOOL:
      info.elem_size_bits = sizeof(int8_t) * 8;
      break;
    case (int)onnx::TensorProto_DataType_UINT16:
    case (int)onnx::TensorProto_DataType_INT16:
    case (int)onnx::TensorProto_DataType_FLOAT16:
    case (int)onnx::TensorProto_DataType_BFLOAT16:
      info.elem_size_bits = sizeof(int16_t) * 8;
      break;
    case (int)onnx::TensorProto_DataType_INT32:
    case (int)onnx::TensorProto_DataType_UINT32:
    case (int)onnx::TensorProto_DataType_FLOAT:
      info.elem_size_bits = sizeof(int32_t) * 8;
      break;
    case (int)onnx::TensorProto_DataType_INT64:
    case (int)onnx::TensorProto_DataType_UINT64:
    case (int)onnx::TensorProto_DataType_DOUBLE:
      info.elem_size_bits = sizeof(int64_t) * 8;
      break;
    case (int)onnx::TensorProto_DataType_UINT4:
    case (int)onnx::TensorProto_DataType_INT4:
      info.elem_size_bits = 4;
      break;
    default:
      ORT_THROW("Unsupported elem type: ", elem_type);
  }
  info.elem_count_all = 1;
  if (const auto* shape_proto = arg.Shape(); shape_proto != nullptr) {
    for (int j = 0; j < shape_proto->dim_size(); ++j) {
      info.shape.push_back(shape_proto->dim(j).dim_value());
      info.elem_count_all *= info.shape.back();
    }
  }
  return info;
}

}  // namespace onnxruntime
