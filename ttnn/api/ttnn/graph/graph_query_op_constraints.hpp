// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tuple>
#include <variant>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/allocator.hpp>

namespace ttnn::graph {

namespace detail {
// Helper to temporarily change logger level
class LogLevelGuard {
public:
    explicit LogLevelGuard(spdlog::level::level_enum new_level) :
        saved_level_(tt::LoggerRegistry::instance().get(tt::LogOp)->level()) {
        tt::LoggerRegistry::instance().set_level(new_level);
    }
    ~LogLevelGuard() { tt::LoggerRegistry::instance().set_level(saved_level_); }
    LogLevelGuard(const LogLevelGuard&) = delete;
    LogLevelGuard& operator=(const LogLevelGuard&) = delete;

private:
    spdlog::level::level_enum saved_level_;
};

// These overloaded extract_output_tensor functions abstract the return type of an arbitrary op from the rest of the
// constraints query function. An overload resolution failure means the return type for the op in that query is not yet
// supported and a new overload should be added

// Generic function to extract all tensors from most return types using reflection. The return type could be:
// - A single Tensor (Most ops)
// - std::vector<Tensor> or std::tuple<Tensor, Tensor, ...> (multi-output ops, e.g., sort,
// split_query_key_value_and_split_heads)
template <typename T>
inline std::vector<Tensor> extract_output_tensors(const T& result) {
    std::vector<Tensor> tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>([&tensors](auto&& tensor) { tensors.push_back(tensor); }, result);
    return tensors;
}

// Specialized overload for conv2d output (std::variant with different tuple combinations)
template <typename... Ts>
inline std::vector<Tensor> extract_output_tensors(const std::variant<Ts...>& result) {
    std::vector<Tensor> tensors;
    std::visit(
        [&tensors](auto&& value) {
            tt::stl::reflection::visit_object_of_type<Tensor>(
                [&tensors](auto&& tensor) { tensors.push_back(tensor); }, value);
        },
        result);
    return tensors;
}

}  // namespace detail

struct ResourceUsage {
    size_t cb_peak_size_per_core = 0;
    size_t l1_buffers_peak_per_core = 0;
    size_t peak_memory_usage_per_core = 0;
    size_t l1_output_buffer_per_core = 0;
};

struct ConstraintQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    ResourceUsage resource_usage;
    std::optional<std::vector<TensorSpec>> output_tensor_specs;
    std::optional<std::string> error_message;
};

/**
 * @brief Captures the graph operations and extracts resource usage constraints.
 *
 * This function runs graph capture by invoking the provided operation with the given arguments,
 * then extracts and returns the resource usage constraints from the captured trace.
 *
 * @tparam Op The type of the operation that will be invoked to capture the graph operations.
 * @tparam Args The types of the arguments that will be passed to the operation.
 * @param op The operation that will be invoked to capture the graph operations.
 * @param device A pointer to the Device object, which provides information about the compute grid size.
 * @param args The arguments that will be passed to the operation.
 * @return ConstraintQueryResponse containing the execution status and resource usage constraints.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_constraints(Op op, tt::tt_metal::distributed::MeshDevice* device, Args&&... args) {
    detail::LogLevelGuard log_guard(spdlog::level::level_enum::off);
    nlohmann::json op_trace;
    std::vector<Tensor> outputs;
    // outer graph capture is to avoid dispatching/allocating dummy input tensors
    {
        auto capture_outer = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);

        // helper lambda to transform TensorSpec to DeviceTensor
        auto transform_arg = [device](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
                return create_device_tensor(arg, device);
            } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, std::optional<TensorSpec>>) {
                return arg ? std::optional<Tensor>(create_device_tensor(*arg, device)) : std::nullopt;
            } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, std::vector<TensorSpec>>) {
                std::vector<Tensor> result(arg.size());
                std::transform(arg.begin(), arg.end(), result.begin(), [device](auto&& item) {
                    return create_device_tensor(item, device);
                });
                return result;
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };
        auto transformed_args = std::make_tuple(transform_arg(std::forward<Args>(args))...);

        // inner graph capture is to capture the actual op graph trace
        try {
            auto capture_inner = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);
            outputs = detail::extract_output_tensors(std::apply(op, transformed_args));
        }  // end of inner graph capture
        catch (const std::exception& e) {
            log_debug(tt::LogOp, "Error during graph capture: {}", e.what());
            return ConstraintQueryResponse{
                ExecutionStatus::Error,
                {.cb_peak_size_per_core = 0,
                 .l1_buffers_peak_per_core = 0,
                 .peak_memory_usage_per_core = 0,
                 .l1_output_buffer_per_core = 0},
                /* output_tensor_specs= */ std::nullopt,
                e.what()};
        }
        op_trace = capture_outer.end_graph_capture();
    }  // end of outer graph capture

    // extract memory footprint from the trace
    auto interleaved_storage_cores = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    const auto& [cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core] =
        extract_resource_usage_per_core(op_trace, interleaved_storage_cores);

    size_t l1_output_buffer_per_core = 0;
    for (const auto& output : outputs) {
        if (!output.buffer()->is_dram()) {
            l1_output_buffer_per_core +=
                extract_l1_output_buffer_allocation_size_per_core(output, interleaved_storage_cores);
        }
    }

    std::vector<TensorSpec> output_specs;
    output_specs.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_specs), [](const Tensor& t) {
        return t.tensor_spec();
    });

    return ConstraintQueryResponse{
        ExecutionStatus::Success,
        {cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core, l1_output_buffer_per_core},
        std::make_optional(std::move(output_specs))};
}

}  // namespace ttnn::graph
