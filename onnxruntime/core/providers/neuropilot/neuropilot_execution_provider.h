
#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

#include <memory>
#include <map>
#include <unordered_map>

namespace onnxruntime {

class NeuroPilotRuntimeApi;
class NeuroPilotRuntimeWrapper;

class NeuropilotExecutionProvider : public IExecutionProvider {
 public:
  explicit NeuropilotExecutionProvider(NeuroPilotRuntimeApi* runtime_api);
  virtual ~NeuropilotExecutionProvider();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeuropilotExecutionProvider);

 public:
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                const IKernelLookup& kernel_lookup,
                                                                const GraphOptimizerRegistry& graph_optimizer_registry,
                                                                IResourceAccountant* resource_accountant = nullptr) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  ProviderOptions GetProviderOptions() const override;

  common::Status SetEpDynamicOptions(gsl::span<const char* const> /*keys*/,
                                     gsl::span<const char* const> /*values*/) override;

  const InlinedVector<const Node*> GetEpContextNodes() const override;

 private:
  NodeComputeInfo init_ep_context_impls(Node& node, const std::string& ep_cache);

 private:
  std::string model_path_;
  NeuroPilotRuntimeApi* runtime_api_;
  std::vector<const Node*> ep_context_nodes_;
  std::map<std::string, std::shared_ptr<NeuroPilotRuntimeWrapper>> dla_handles_;
  std::unordered_map<std::string, std::string> options_;
};

}  // namespace onnxruntime
