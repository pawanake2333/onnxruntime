
#include "neuropilot_execution_provider.h"

#include <vector>
#include <memory>
#include <cinttypes>

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/run_options.h"
#include "core/providers/partitioning_utils.h"

#include "np_util.h"
#include "runtime_api.h"

namespace onnxruntime {

static const std::string EP_EPCONTEXT_OP = "EPContext";
static const std::string EP_SOURCE_ATTR = "source";
static const std::string EP_SOURCE_NEUROPILOT = "Neuropilot";

NeuropilotExecutionProvider::NeuropilotExecutionProvider(NeuroPilotRuntimeApi* runtime_api) : IExecutionProvider(onnxruntime::kNeuropilotExecutionProvider), runtime_api_(runtime_api) {
  for (const auto& [key, values] : runtime_api_->getConfigEntryValues()) {
    options_[std::string(key)] = values[0];
  }
  for (const auto& [key, values] : runtime_api_->getConfigEntryRanges()) {
    options_[std::string(key)] = std::to_string(std::get<2>(values));
  }
}

NeuropilotExecutionProvider::~NeuropilotExecutionProvider() {
  ep_context_nodes_.clear();
  dla_handles_.clear();
}

std::shared_ptr<KernelRegistry> getRegister() {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

std::shared_ptr<KernelRegistry> NeuropilotExecutionProvider::GetKernelRegistry() const {
  return getRegister();
}

std::vector<std::unique_ptr<ComputeCapability>> NeuropilotExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                                           const IKernelLookup& kernel_lookup,
                                                                                           const GraphOptimizerRegistry& graph_optimizer_registry,
                                                                                           IResourceAccountant* resource_accountant) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;
  if (graph_viewer.IsSubgraph()) {
    return capabilities;
  }
  std::shared_ptr<int64_t> node_index = std::make_shared<int64_t>(0);
  auto gen_metadef_name = [node_index]() {
    char meta_name[128];
    snprintf(meta_name, sizeof(meta_name), "EpNode_%" PRId64, (*node_index)++);
    return std::string(meta_name);
  };
  for (const auto& node : graph_viewer.Nodes()) {
    if (node.OpType() != EP_EPCONTEXT_OP) {
      continue;
    }
    auto& attrs = node.GetAttributes();
    if (auto it = attrs.find(EP_SOURCE_ATTR); it == attrs.end() || it->second.s() != EP_SOURCE_NEUROPILOT) {
      ORT_THROW("EPContext node for Neuropilot does not support this node(", node.Name(), ").");
    }
    std::vector<const Node*> nodes = {&node};
    capabilities.push_back(utils::MakeComputeCapability(graph_viewer, nodes, gen_metadef_name, "Neuropilot", false));
  }
  return capabilities;
}

common::Status NeuropilotExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                    std::vector<NodeComputeInfo>& node_compute_funcs) {
  if (!fused_nodes_and_graphs.empty()) {
    const onnxruntime::GraphViewer& tmp_viewer(fused_nodes_and_graphs[0].filtered_graph);
    model_path_ = std::filesystem::path(tmp_viewer.ModelPath().native()).parent_path();
  }
  for (auto& inode : fused_nodes_and_graphs) {
    auto& fused_node = inode.fused_node.get();
    std::string ep_cache;
    for (const auto& rnode : inode.filtered_graph.get().Nodes()) {
      if (rnode.OpType() != EP_EPCONTEXT_OP || !ep_cache.empty()) {
        LOGS(*GetLogger(), WARNING) << "fused node with invalide node " << rnode.Name() << ", skipped";
        continue;
      }
      auto& attrs = rnode.GetAttributes();
      if (auto it = attrs.find("ep_cache_context"); it == attrs.end() || it->second.type() != onnx::AttributeProto::STRING) {
        LOGS(*GetLogger(), ERROR) << "ep_cache_context attribute is required for EPContext node (" << rnode.Name() << ").";
        ORT_THROW("ep_cache_context attribute is required for EPContext node (", rnode.Name(), ").");
        continue;
      } else {
        ep_cache = it->second.s();
      }
    }
    auto compute_info = init_ep_context_impls(fused_node, ep_cache);
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

ProviderOptions NeuropilotExecutionProvider::GetProviderOptions() const {
  return options_;
}

common::Status NeuropilotExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                                gsl::span<const char* const> values) {
  if (keys.size() != values.size()) {
    return Status::OK();
  }

  std::unordered_map<std::string, int32_t> updated_values;
  const auto& entry_values = runtime_api_->getConfigEntryValues();
  for (size_t i = 0; i < keys.size(); ++i) {
    auto iter = entry_values.find(keys[i]);
    if (iter == entry_values.end()) {
      continue;
    }
    auto val_pos = std::find(iter->second.begin(), iter->second.end(), values[i]);
    if (val_pos == iter->second.end()) {
      return common::Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "invalid ep options.");
    }
    updated_values[keys[i]] = (int32_t)(val_pos - iter->second.begin());
    options_[std::string(keys[i])] = values[i];
  }

  const auto& entry_ranges = runtime_api_->getConfigEntryRanges();
  for (size_t i = 0; i < keys.size(); ++i) {
    auto iter = entry_ranges.find(keys[i]);
    if (iter == entry_ranges.end()) {
      continue;
    }
    auto ivalue = atoi(values[i]);
    if (ivalue < std::get<0>(iter->second) || ivalue > std::get<1>(iter->second)) {
      return common::Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "invalid ep options.");
    }
    updated_values[keys[i]] = ivalue;
    options_[std::string(keys[i])] = std::to_string(ivalue);
  }
  if (!updated_values.empty()) {
    return Status::OK();
  }
  for (auto& [_, handle] : dla_handles_) {
    for (auto& [key, value] : updated_values) {
      handle->setConfig(key, value);
    }
  }
  return Status::OK();
}

const InlinedVector<const Node*> NeuropilotExecutionProvider::GetEpContextNodes() const {
  return InlinedVector<const Node*>{ep_context_nodes_.begin(), ep_context_nodes_.end()};
}

NodeComputeInfo NeuropilotExecutionProvider::init_ep_context_impls(Node& node, const std::string& ep_cache) {
  if (ep_cache.empty() || ep_cache[0] == '.' || ep_cache.find('/') != std::string::npos || ep_cache.find('\\') != std::string::npos) {
    LOGS(*GetLogger(), ERROR) << "init Neuropilot context node: " << node.Name() << ", invalid ep_cache path: " << ep_cache;
    ORT_THROW("invalid ep_cache file name.");
  }
  auto ep_cache_file = std::filesystem::path(model_path_) / ep_cache;
  if (!std::filesystem::exists(ep_cache_file)) {
    LOGS(*GetLogger(), ERROR) << "init Neuropilot context node: " << node.Name() << ", ep_cache not exists: " << ep_cache;
    ORT_THROW("ep_cache file does not exist (", ep_cache_file.string(), ").");
  }

  std::vector<ArgShapeInfo> input_infos, output_infos;
  for (auto& iarg : node.InputDefs()) {
    input_infos.push_back(get_arg_shape_size(*iarg));
  }
  for (auto& oarg : node.OutputDefs()) {
    output_infos.push_back(get_arg_shape_size(*oarg));
  }

#if defined(ANDROID)
  std::shared_ptr<NeuroPilotRuntimeWrapper> dla_runtime = std::make_shared<NeuroPilotRuntimeWrapper>(
      ep_cache_file.string(),
      runtime_api_,
      std::move(input_infos),
      std::move(output_infos));
  std::weak_ptr<NeuroPilotRuntimeWrapper> weak_runtime = dla_runtime;

  NodeComputeInfo compute_info;
  compute_info.create_state_func = [](ComputeContext* context, FunctionState* state) { return 0; };
  compute_info.release_state_func = [](FunctionState state) {};
  compute_info.compute_func = [weak_runtime](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
    if (auto dla_runtime = weak_runtime.lock(); dla_runtime && dla_runtime->isValid()) {
      if (auto err = dla_runtime->run(context); !err.empty()) {
        return common::Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, err);
      }
      return common::Status::OK();
    } else {
      return common::Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "internal neuropilot runtime has already been closed.");
    }
  };

  ep_context_nodes_.push_back(&node);
  dla_handles_[node.Name()] = dla_runtime;
#else
  NodeComputeInfo compute_info;
  compute_info.create_state_func = [](ComputeContext* context, FunctionState* state) { return 0; };
  compute_info.release_state_func = [](FunctionState state) {};
  compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
    return common::Status::OK();
  };
#endif
  LOGS(*GetLogger(), INFO) << "init Neuropilot context node:" << node.Name() << ", load ep_cache: " << ep_cache << " success.";
  return compute_info;
}

}  // namespace onnxruntime
