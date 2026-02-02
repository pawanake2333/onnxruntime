
#include "runtime_api.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"

#include "onnxruntime_cxx_api.h"

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

namespace onnxruntime {

std::string_view kNpOpt_Preference = "preference";
std::string_view kNpOpt_Preference_Values[] = {
    STRINGIFY(NEURONRUNTIME_PREFER_PERFORMANCE),
    STRINGIFY(NEURONRUNTIME_PREFER_POWER),
    STRINGIFY(NEURONRUNTIME_PREFER_BOOST),
};

std::string_view kNpOpt_Priority = "priority";
std::string_view kNpOpt_Priority_Values[] = {
    STRINGIFY(NEURONRUNTIME_PRIORITY_LOW),
    STRINGIFY(NEURONRUNTIME_PRIORITY_MEDIUM),
    STRINGIFY(NEURONRUNTIME_PRIORITY_HIGH),
};

std::string_view kNpOpt_BoostValue = "boostValue";
std::string_view kNpOpt_BoostValue_Values[] = {
    STRINGIFY(NEURONRUNTIME_BOOSTVALUE_MIN),
};

std::string_view kNpOpt_PowerPolicy = "powerPolicy";
std::string_view kNpOpt_PowerPolicy_Values[] = {
    STRINGIFY(NEURONRUNTIME_POWER_POLICY_DEFAULT),
    STRINGIFY(NEURONRUNTIME_POWER_POLICY_SUSTAINABLE),
    STRINGIFY(NEURONRUNTIME_POWER_POLICY_PERFORMANCE),
    STRINGIFY(NEURONRUNTIME_POWER_POLICY_POWER_SAVING),
};

NeuroPilotRuntimeApi::NeuroPilotRuntimeApi(const std::string& base) : DymLoader(base, RTLD_NOW) {
  LoadSym(NeuronRuntimeV2_create);
  LoadSym(NeuronRuntimeV2_release);
  LoadSym(NeuronRuntimeV2_create_with_options);
  LoadSym(NeuronRuntimeV2_createFromBuffer);
  LoadSym(NeuronRuntimeV2_createFromBuffer_with_options);
  LoadSym(NeuronRuntimeV2_enqueue);
  LoadSym(NeuronRuntimeV2_run);

  LoadSym(NeuronRuntimeV2_getInputNumber);
  LoadSym(NeuronRuntimeV2_getInputRank);
  LoadSym(NeuronRuntimeV2_getInputSize);
  LoadSym(NeuronRuntimeV2_getInputPaddedSize);
  LoadSym(NeuronRuntimeV2_getInputPaddedDimensions);
  LoadSym(NeuronRuntimeV2_getInputPitch);
  LoadSym(NeuronRuntimeV2_setInputShape);

  LoadSym(NeuronRuntimeV2_getOutputNumber);
  LoadSym(NeuronRuntimeV2_getOutputSize);
  LoadSym(NeuronRuntimeV2_getOutputPaddedSize);
  LoadSym(NeuronRuntimeV2_getOutputPaddedDimensions);
  LoadSym(NeuronRuntimeV2_getOutputPitch);
  LoadSym(NeuronRuntimeV2_getOutputRank);

  LoadSym(NeuronRuntimeV2_getProfiledQoSData);
  LoadSym(NeuronRuntimeV2_getMetadataInfo);
  LoadSym(NeuronRuntimeV2_getMetadata);
  LoadSym(NeuronRuntimeV2_setQoS);
  LoadSym(NeuronRuntimeV2_setQoSOption);
}

NeuroPilotRuntimeApi* NeuroPilotRuntimeApi::getInstance(const std::string& base) {
  static std::mutex lock;
  static NeuroPilotRuntimeApi* instance = nullptr;
  if (instance == nullptr) {
    std::lock_guard<std::mutex> guard(lock);
    if (instance == nullptr) {
      instance = new NeuroPilotRuntimeApi(base);
    }
  }
  return instance;
}

const std::unordered_map<std::string_view, gsl::span<std::string_view>>& NeuroPilotRuntimeApi::getConfigEntryValues() {
  static const std::unordered_map<std::string_view, gsl::span<std::string_view>> config_entry_values = {
      {kNpOpt_Preference, kNpOpt_Preference_Values},
      {kNpOpt_Priority, kNpOpt_Priority_Values},
      {kNpOpt_PowerPolicy, kNpOpt_PowerPolicy_Values},
  };
  return config_entry_values;
}

const std::unordered_map<std::string_view, std::tuple<int32_t, int32_t, int32_t>>& NeuroPilotRuntimeApi::getConfigEntryRanges() {
  static const std::unordered_map<std::string_view, std::tuple<int32_t, int32_t, int32_t>> config_entry_range_values = {
      {kNpOpt_BoostValue, {NEURONRUNTIME_BOOSTVALUE_MIN, NEURONRUNTIME_BOOSTVALUE_MAX, NEURONRUNTIME_BOOSTVALUE_MIN}},
  };
  return config_entry_range_values;
}

static void initQosOption(QoSOptions& qos_config) {
  qos_config.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
  qos_config.priority = NEURONRUNTIME_PRIORITY_LOW;
  qos_config.boostValue = NEURONRUNTIME_BOOSTVALUE_MIN;
  qos_config.maxBoostValue = NEURONRUNTIME_BOOSTVALUE_MAX;
  qos_config.minBoostValue = NEURONRUNTIME_BOOSTVALUE_MIN;
  qos_config.deadline = 0;
  qos_config.abortTime = 0;
  qos_config.delayedPowerOffTime = NEURONRUNTIME_POWER_OFF_TIME_DEFAULT;
  qos_config.powerPolicy = NEURONRUNTIME_POWER_POLICY_DEFAULT;
  qos_config.applicationType = NEURONRUNTIME_APP_NORMAL;
  qos_config.profiledQoSData = nullptr;
}

NeuroPilotRuntimeWrapper::NeuroPilotRuntimeWrapper(const std::string& model_path, NeuroPilotRuntimeApi* api, std::vector<ArgShapeInfo>&& input_infos, std::vector<ArgShapeInfo>&& output_infos, size_t backlogs)
    : api_(api), handler_(nullptr), _qos_dirty(false), input_infos_(std::move(input_infos)), output_infos_(std::move(output_infos)) {
  if (int r = api_->NeuronRuntimeV2_create(model_path.c_str(), 4, &handler_, backlogs); r != 0) {
    ORT_THROW("create model failed, err=", r);
  }
  initQosOption(_qos_config);
  _qos_dirty = true;
}

NeuroPilotRuntimeWrapper::~NeuroPilotRuntimeWrapper() {
  close();
}

void NeuroPilotRuntimeWrapper::close() {
  if (!handler_) {
    return;
  }
  api_->NeuronRuntimeV2_release(handler_);
  handler_ = nullptr;
}

std::string NeuroPilotRuntimeWrapper::run(OrtKernelContext* ctx_) {
  Ort::KernelContext ctx(ctx_);
  if (!handler_) {
    ORT_THROW("neuropilot runtime handle has already been closed.");
  }
  std::vector<IOBuffer> inputs, outputs;
  for (size_t i = 0; i < input_infos_.size(); i++) {
    auto arg = ctx.GetInput(i);
    inputs.push_back({(void*)arg.GetTensorRawData(), arg.GetTensorSizeInBytes(), -1});
  }
  for (size_t i = 0; i < output_infos_.size(); i++) {
    Ort::UnownedValue output_tensor = ctx.GetOutput(i, output_infos_[i].shape);
    outputs.push_back({(void*)output_tensor.GetTensorMutableData<void>(), (size_t)(output_infos_[i].elem_count_all * output_infos_[i].elem_size_bits / 8), -1});
  }
  if (_qos_dirty && _qos_lock.try_lock()) {
    // README: set qos option when model is inferencing will cause undefined behavior
    api_->NeuronRuntimeV2_setQoSOption(handler_, &_qos_config);
    _qos_dirty = false;
    _qos_lock.unlock();
  }
  std::shared_lock guard{_qos_lock};
  int ret = api_->NeuronRuntimeV2_run(handler_, SyncInferenceRequest{inputs.data(), outputs.data()});
  if (ret != 0) {
    return "run failed, err=" + std::to_string(ret);
  }
  return "";
}

void NeuroPilotRuntimeWrapper::setConfig(const std::string& key, int32_t value) {
  if (key == kNpOpt_BoostValue) {
    _qos_config.boostValue = (uint8_t)value;
    _qos_dirty = true;
  } else if (key == kNpOpt_Preference) {
    _qos_config.preference = (RuntimeAPIQoSPreference)value;
    _qos_dirty = true;
  } else if (key == kNpOpt_Priority) {
    _qos_config.priority = (RuntimeAPIQoSPriority)value;
    _qos_dirty = true;
  } else if (key == kNpOpt_PowerPolicy) {
    _qos_config.powerPolicy = (RuntimeAPIQoSPowerPolicy)value;
    _qos_dirty = true;
  }
}

}  // namespace onnxruntime
