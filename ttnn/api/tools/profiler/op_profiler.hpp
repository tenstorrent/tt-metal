// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lightweight device-operation path — safe for wide include chains:
#include "tools/profiler/op_profiler_serialize.hpp"

// Heavy includes below are only needed for the ExternalOperation path
// (used in pytensor.cpp and similar .cpp files, NOT in device_operation.hpp).
#include <algorithm>
#include <limits>
#include <optional>
#include <stack>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

using json = nlohmann::json;

namespace tt::tt_metal::op_profiler {

#if defined(TRACY_ENABLE)

static inline json get_tensor_json(const Tensor& tensor) {
    json ret;
    if (tensor.storage_type() == StorageType::DEVICE) {
        ret["storage_type"]["device_id"] = tensor.device()->id();
        ret["storage_type"]["memory_config"]["buffer_type"] =
            enchantum::to_string(tensor.memory_config().buffer_type());
        ret["storage_type"]["memory_config"]["memory_layout"] =
            enchantum::to_string(tensor.memory_config().memory_layout());
    } else {
        ret["storage_type"] = fmt::format("{}", enchantum::to_string(tensor.storage_type()));
    }

    auto tensor_shape_padded = tensor.padded_shape();
    auto tensor_shape_logical = tensor.logical_shape();
    ret["shape"]["W"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 4 ? tensor_shape_padded[-4] : 1,
        tensor_shape_logical.rank() >= 4 ? tensor_shape_logical[-4] : 1);
    ret["shape"]["Z"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 3 ? tensor_shape_padded[-3] : 1,
        tensor_shape_logical.rank() >= 3 ? tensor_shape_logical[-3] : 1);
    ret["shape"]["Y"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 2 ? tensor_shape_padded[-2] : 1,
        tensor_shape_logical.rank() >= 2 ? tensor_shape_logical[-2] : 1);
    ret["shape"]["X"] = fmt::format("{}[{}]", tensor_shape_padded[-1], tensor_shape_logical[-1]);
    ret["layout"] = fmt::format("{}", enchantum::to_string(tensor.layout()));
    ret["dtype"] = fmt::format("{}", enchantum::to_string(tensor.dtype()));

    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<Tensor>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    ret.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        ret.push_back(get_tensor_json(tensor));
    }
    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<std::optional<const Tensor>>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    for (const auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<std::optional<Tensor>>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    for (const auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

template <bool IsExternal = false, typename Operation>
inline json get_base_json(
    uint32_t opID,
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    std::optional<std::reference_wrapper<typename Operation::OutputTensors>> output_tensors = std::nullopt) {
    ZoneScoped;
    json j;
    j["global_call_count"] = opID;

    std::string opName = op.get_type_name();

    if constexpr (!IsExternal) {
        auto profiler_info = op.create_profiler_info(input_tensors);
        if (profiler_info.preferred_name.has_value()) {
            j["op_code"] = profiler_info.preferred_name.value();
        }

        if (profiler_info.parallelization_strategy.has_value()) {
            j["parallelization_strategy"] = profiler_info.parallelization_strategy.value();
        }
    }

    std::replace(opName.begin(), opName.end(), ',', ';');
    j["op_code"] = opName;

    json attributesObj;
    auto attributes = op.attributes();
    if (not attributes.empty()) {
        ZoneScopedN("get_attributes_json");
        for (auto&& [name, value] : attributes) {
            std::string nameStr;
            nameStr = fmt::format("{}", name);
            attributesObj[nameStr] = fmt::format("{}", value);
        }
    }

    j["attributes"] = attributesObj;

    j["input_tensors"] = get_tensors_json(input_tensors);

    if (output_tensors.has_value()) {
        j["output_tensors"] = get_tensors_json(output_tensors.value());
    }
    return j;
}

#endif  // TRACY_ENABLE

inline std::string op_meta_data_serialized_json(
    uint32_t opID, const tt::tt_metal::operation::ExternalOperation& op, const std::vector<Tensor>& input_tensors) {
#if defined(TRACY_ENABLE)
    if (!is_op_profiler_env_var_set()) {
        return {};
    }
    auto j = get_base_json<true>(opID, op, input_tensors);
    j["op_type"] = enchantum::to_string(OpType::python_fallback);
    std::string ser = j.dump(4);
    return fmt::format("`TT_DNN_FALL_BACK_OP:{} ->\n{}`", j["op_code"].dump(), ser);
#else
    return {};
#endif
}

#if defined(TRACY_ENABLE)

#define TracyOpTTNNDevice(                                                                                        \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value, program_cache_hit)          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                \
        std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                         \
            operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value, program_cache_hit); \
        std::string op_text = fmt::format("id:{}", operation_id);                                                 \
        ZoneText(op_text.c_str(), op_text.size());                                                                \
        tt::tt_metal::op_profiler::tracy_message(op_message);                                                     \
    }

#define TracyOpTTNNExternal(op, input_tensors, base_op_id)                                                          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                  \
        /* This op runs entirely on host, but its ID must be generated using the same data-path as device-side */   \
        /* ops, for accurate reporting by the performance post-processor. */                                        \
        auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(base_op_id, 0, true);                           \
        std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(op_id, op, input_tensors); \
        std::string op_text = fmt::format("id:{}", op_id);                                                          \
        ZoneText(op_text.c_str(), op_text.size());                                                                  \
        tt::tt_metal::op_profiler::tracy_message(op_message);                                                       \
    }

#else

#define TracyOpTTNNDevice( \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value, program_cache_hit)
#define TracyOpTTNNExternal(op, input_tensors, base_op_id)

#endif

}  // namespace tt::tt_metal::op_profiler
