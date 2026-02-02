

if(NOT ANDROID)
    #message(FATAL_ERROR "NEUROPILOT is not supported on non-Android platforms.")
endif()

message(STATUS "enabling onnxruntime_providers_neuropilot.")
message(STATUS "NEUROPILOT_SDK_ROOT: ${NEUROPILOT_SDK_ROOT}")

if(NOT IS_DIRECTORY "${NEUROPILOT_SDK_ROOT}")
    message(FATAL_ERROR "NEUROPILOT_SDK_ROOT is not a valid directory: ${NEUROPILOT_SDK_ROOT}")
endif()

file(GLOB_RECURSE
    onnxruntime_providers_neuropilot_ep_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/neuropilot/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/neuropilot/*.cc"
)

onnxruntime_add_static_library(onnxruntime_providers_neuropilot ${onnxruntime_providers_neuropilot_ep_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_neuropilot
    onnxruntime_common onnxruntime_framework onnx
    onnx_proto protobuf::libprotobuf-lite
    flatbuffers::flatbuffers Boost::mp11
    nlohmann_json::nlohmann_json
)
target_include_directories(onnxruntime_providers_neuropilot
    PRIVATE ${NEUROPILOT_SDK_ROOT}/usdk/include
)
set_target_properties(onnxruntime_providers_neuropilot PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_neuropilot PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_neuropilot PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS onnxruntime_providers_neuropilot EXPORT ${PROJECT_NAME}Targets
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
