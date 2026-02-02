#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <neuron/api/RuntimeV2.h>

#include "np_util.h"

struct OrtKernelContext;

namespace onnxruntime {

class NeuroPilotRuntimeApi : public DymLoader {
 private:
  NeuroPilotRuntimeApi(const std::string& base);
  NeuroPilotRuntimeApi(const NeuroPilotRuntimeApi&) = delete;
  NeuroPilotRuntimeApi& operator=(const NeuroPilotRuntimeApi&) = delete;

 public:
  static NeuroPilotRuntimeApi* getInstance(const std::string& base);
  static const std::unordered_map<std::string_view, gsl::span<std::string_view>>& getConfigEntryValues();
  static const std::unordered_map<std::string_view, std::tuple<int32_t, int32_t, int32_t>>& getConfigEntryRanges();

 public:
  DeclareSym(NeuronRuntimeV2_create);
  DeclareSym(NeuronRuntimeV2_release);
  DeclareSym(NeuronRuntimeV2_create_with_options);
  DeclareSym(NeuronRuntimeV2_createFromBuffer);
  DeclareSym(NeuronRuntimeV2_createFromBuffer_with_options);
  DeclareSym(NeuronRuntimeV2_enqueue);
  DeclareSym(NeuronRuntimeV2_run);

  DeclareSym(NeuronRuntimeV2_getInputNumber);
  DeclareSym(NeuronRuntimeV2_getInputRank);
  DeclareSym(NeuronRuntimeV2_getInputSize);
  DeclareSym(NeuronRuntimeV2_getInputPaddedSize);
  DeclareSym(NeuronRuntimeV2_getInputPaddedDimensions);
  DeclareSym(NeuronRuntimeV2_getInputPitch);
  DeclareSym(NeuronRuntimeV2_setInputShape);

  DeclareSym(NeuronRuntimeV2_getOutputNumber);
  DeclareSym(NeuronRuntimeV2_getOutputSize);
  DeclareSym(NeuronRuntimeV2_getOutputPaddedSize);
  DeclareSym(NeuronRuntimeV2_getOutputPaddedDimensions);
  DeclareSym(NeuronRuntimeV2_getOutputPitch);
  DeclareSym(NeuronRuntimeV2_getOutputRank);

  DeclareSym(NeuronRuntimeV2_getProfiledQoSData);
  DeclareSym(NeuronRuntimeV2_getMetadataInfo);
  DeclareSym(NeuronRuntimeV2_getMetadata);
  DeclareSym(NeuronRuntimeV2_setQoS);
  DeclareSym(NeuronRuntimeV2_setQoSOption);
};

class NeuroPilotRuntimeWrapper {
 private:
  NeuroPilotRuntimeApi* api_;
  void* handler_;
  QoSOptions _qos_config;
  std::shared_mutex _qos_lock;
  std::atomic_bool _qos_dirty;
  std::vector<ArgShapeInfo> input_infos_;
  std::vector<ArgShapeInfo> output_infos_;

 private:
  NeuroPilotRuntimeWrapper(const NeuroPilotRuntimeWrapper&) = delete;
  NeuroPilotRuntimeWrapper& operator=(const NeuroPilotRuntimeWrapper&) = delete;

 public:
  NeuroPilotRuntimeWrapper(const std::string& model_path, NeuroPilotRuntimeApi* api, std::vector<ArgShapeInfo>&& input_infos, std::vector<ArgShapeInfo>&& output_infos, size_t nb_threads = 4);
  ~NeuroPilotRuntimeWrapper();

 public:
  std::string run(OrtKernelContext* ctx);
  void close();
  inline bool isValid() const { return handler_ != nullptr; }
  void setConfig(const std::string& key, int32_t value);
};

}  // namespace onnxruntime
