

#include "core/providers/neuropilot/neuropilot_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "onnxruntime_cxx_api.h"
#include "neuropilot_provider_factory_creator.h"
#include "neuropilot_execution_provider.h"
#include "runtime_api.h"

namespace onnxruntime {

static const char* kNeuropilotRuntimePath = "runtime_path";

struct NeuropilotProviderFactory : IExecutionProviderFactory {
 private:
  std::string runtime_library_path_;
  NeuroPilotRuntimeApi* runtime_api_;

 public:
  explicit NeuropilotProviderFactory(const ProviderOptions& provider_options_map);
  virtual ~NeuropilotProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options, const OrtLogger& session_logger) override;
};

NeuropilotProviderFactory::NeuropilotProviderFactory(const ProviderOptions& provider_options_map) {
  if (auto it = provider_options_map.find(kNeuropilotRuntimePath); it != provider_options_map.end()) {
    runtime_library_path_ = it->second;
  }
#if !defined(ANDROID)
  return;
#endif
  if (!support_neuropilot()) {
    ORT_THROW("NeuropilotProviderFactory: neuropilot is not supported on this platform.");
    return;
  }
  if (runtime_api_ = NeuroPilotRuntimeApi::getInstance(runtime_library_path_); runtime_api_ == nullptr) {
    ORT_THROW("NeuropilotProviderFactory: failed to get neuropilot runtime api.");
    return;
  }
}

std::unique_ptr<IExecutionProvider> NeuropilotProviderFactory::CreateProvider() {
  return std::make_unique<NeuropilotExecutionProvider>(runtime_api_);
}

std::unique_ptr<IExecutionProvider> NeuropilotProviderFactory::CreateProvider(const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
  auto ep = std::make_unique<NeuropilotExecutionProvider>(runtime_api_);
  ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
  return ep;
}

std::shared_ptr<IExecutionProviderFactory> NeuropilotProviderFactoryCreator::Create(const ProviderOptions& provider_options_map, const SessionOptions* session_options) {
  auto ep = std::make_shared<NeuropilotProviderFactory>(provider_options_map);
  return ep;
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Neuropilot, _In_ OrtSessionOptions* options) {
  Ort::SessionOptions optImpl(options);
  onnxruntime::ProviderOptions provider_options_map = {
      {onnxruntime::kNeuropilotRuntimePath, optImpl.GetConfigEntryOrDefault(onnxruntime::kNeuropilotRuntimePath, "neuropilot_runtime.mtk.so")},
  };
  options->provider_factories.push_back(onnxruntime::NeuropilotProviderFactoryCreator::Create(provider_options_map, nullptr));
  return nullptr;
}
