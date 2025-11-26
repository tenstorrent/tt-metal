// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tracy/Tracy.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using json = nlohmann::json;

namespace tt::tt_metal::op_profiler {

enum class OpType { python_fallback, tt_dnn_cpu, tt_dnn_device, unknown };
class RuntimeIDToOpName {
    using RuntimeID = uint32_t;
    using KeyType = std::pair<ChipId, RuntimeID>;
    using MapType = std::map<KeyType, std::string>;

public:
    MapType::iterator find(ChipId device_id, RuntimeID runtime_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.find({device_id, runtime_id});
    }
    std::string at(ChipId device_id, RuntimeID runtime_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.at({device_id, runtime_id});
    }
    void insert(KeyType key, std::string opname) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        map.emplace(key, std::move(opname));
    }
    MapType export_map() {
        // thread-safe copy of internal map contents
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map;
    }

private:
    std::mutex map_mutex;
    MapType map;
};

inline RuntimeIDToOpName runtime_id_to_opname_{};

void start_tracy_zone(const std::string& source, const std::string& functName, uint32_t lineNum, uint32_t color = 0);

bool stop_tracy_zone(const std::string& name = "", uint32_t color = 0);

constexpr auto tracy_max_message_length =
    static_cast<size_t>(std::numeric_limits<uint16_t>::max());  // Tracy hard limit is 64KiB including null terminator

void tracy_message(const std::string& source, uint32_t color = 0xf0f8ff);

void tracy_frame();

#if defined(TRACY_ENABLE)
std::string op_meta_data_serialized_json(
    uint32_t opID, const tt::tt_metal::operation::ExternalOperation& op, const std::vector<Tensor>& input_tensors);

template <typename device_operation_t>
std::string op_meta_data_serialized_json(
    const device_operation_t& operation,
    uint32_t operation_id,
    auto device_id,
    const auto& program,
    const auto& operation_attributes,
    const auto& tensor_args,
    auto& tensor_return_value);

bool is_op_profiler_env_var_set();

#define TracyOpTTNNDevice(                                                                                        \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value)          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                \
        std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                         \
            operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value); \
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

#define TracyOpMeshWorkload(                                                                                       \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value)                 \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                 \
        for (const auto& [range, program] : (mesh_workload).get_programs()) {                                      \
            auto base_program_id = program.get_runtime_id();                                                       \
            for (auto coord : range) {                                                                             \
                /* Important! `TT_DNN_DEVICE_OP` must be used in conjunction with `TracyOpMeshWorkload` to feed */ \
                /* regression tests well-formed data. */                                                           \
                /* TODO: (Issue #20233): Move the zone below outside TracyOpMeshWorkload. */                       \
                if (!(mesh_device)->is_local(coord)) {                                                             \
                    continue;                                                                                      \
                }                                                                                                  \
                ZoneScopedN("TT_DNN_DEVICE_OP");                                                                   \
                auto device_id = (mesh_device)->get_device(coord)->id();                                           \
                auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(base_program_id, device_id);           \
                std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                  \
                    operation, op_id, device_id, program, operation_attributes, tensor_args, tensor_return_value); \
                std::string op_text = fmt::format("id:{}", op_id);                                                 \
                ZoneText(op_text.c_str(), op_text.size());                                                         \
                tt::tt_metal::op_profiler::tracy_message(op_message);                                              \
            }                                                                                                      \
        }                                                                                                          \
    }

#else

#define TracyOpTTNNDevice( \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value)
#define TracyOpTTNNExternal(op, input_tensors, base_op_id)
#define TracyOpMeshWorkload( \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value)

#endif
}  // namespace tt::tt_metal::op_profiler
