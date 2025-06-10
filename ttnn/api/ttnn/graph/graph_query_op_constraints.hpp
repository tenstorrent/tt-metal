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

namespace ttnn::graph {

namespace detail {

// These overloaded extract_output_tensor functions abstract the return type of an arbitrary op from the rest of the
// constraints query function. An overload resolution failure means the return type for the op in that query is not yet
// supported and a new overload should be added

// most ops just return a tensor
inline Tensor extract_output_tensor(const Tensor& result) { return result; }

// conv2d output
template <typename... Args1, typename... Args2>
Tensor extract_output_tensor(const std::variant<
                             ttnn::Tensor,
                             std::tuple<ttnn::Tensor, Args1...>,
                             std::tuple<ttnn::Tensor, Args2...>,
                             std::tuple<ttnn::Tensor, Args1..., Args2...>>& result) {
    return std::visit<Tensor>(
        tt::stl::overloaded{
            [](const ttnn::Tensor& arg) { return arg; }, [](const auto& arg) { return std::get<0>(arg); }},
        result);
}
}  // namespace detail

struct ResourceUsage {
    size_t cb_peak_size_per_core = 0;
    size_t l1_buffers_peak_per_core = 0;
    size_t l1_output_buffer_per_core = 0;
};

struct ConstraintQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    ResourceUsage resource_usage;
    std::optional<TensorSpec> output_tensor_spec;
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
auto query_op_constraints(Op op, IDevice* device, Args&&... args) {
    nlohmann::json op_trace;
    Tensor output;
    // outer graph capture is to avoid dispatching/allocating dummy input tensors
    {
        auto capture_outer = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);

        // helper lambda to transform TensorSpec to DeviceTensor
        auto transform_arg = [device](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
                return create_device_tensor(arg, device);
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
            output = detail::extract_output_tensor(std::apply(op, transformed_args));
            op_trace = capture_inner.end_graph_capture();
        }  // end of inner graph capture
        catch (const std::exception& e) {
            log_debug(tt::LogOp, "Error during graph capture: {}", e.what());
            return ConstraintQueryResponse{
                ExecutionStatus::Error, {0, 0, 0}, /* output_tensor_spec= */ std::nullopt, e.what()};
        }

    }  // end of outer graph capture

    // extract memory footprint from the trace
    auto interleaved_storage_cores = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    size_t cb_peak_size_per_core = extract_circular_buffers_peak_size_per_core(op_trace);
    size_t l1_buffers_peak_per_core =
        extract_l1_buffer_allocation_peak_size_per_core(op_trace, interleaved_storage_cores);
    size_t l1_output_buffer_per_core = output.buffer()->is_dram() ? 0
                                                                  : extract_l1_output_buffer_allocation_size_per_core(
                                                                        output, interleaved_storage_cores);

    return ConstraintQueryResponse{
        ExecutionStatus::Success,
        {cb_peak_size_per_core, l1_buffers_peak_per_core, l1_output_buffer_per_core},
        output.tensor_spec()};
}

}  // namespace ttnn::graph
