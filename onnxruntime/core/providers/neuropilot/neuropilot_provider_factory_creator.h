

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
struct SessionOptions;

struct NeuropilotProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions& provider_options_map, const SessionOptions* session_options);
};

}  // namespace onnxruntime
