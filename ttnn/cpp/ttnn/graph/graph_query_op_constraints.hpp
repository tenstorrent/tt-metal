// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttnn::graph {

enum class ExecutionStatus { Success, Error };

struct ResourceUsage {
    size_t cb_peak_size_per_core = 0;
    size_t l1_buffers_peak_per_core = 0;
    size_t l1_output_buffer_per_core = 0;
};

struct QueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    ResourceUsage resource_usage;
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
 * @return QueryResponse containing the execution status and resource usage constraints.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_constraints(Op op, IDevice* device, Args&&... args) {
    uint32_t num_of_active_graph_captures = 0;
    try {
        nlohmann::json op_trace;
        // outer graph capture is to avoid dispatching/allocating dummy input tensors
        {
            auto capture_outer = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);

            // helper lambda to transform TensorSpec to DeviceTensor
            auto transform_arg = [device](auto&& arg) {
                if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
                    return create_device_tensor(arg, device);
                } else {
                    return std::forward<decltype(arg)>(arg);
                }
            };
            auto transformed_args = std::make_tuple(transform_arg(std::forward<Args>(args))...);

            // inner graph capture is to capture the actual op graph trace
            {
                auto capture_inner = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);
                std::apply(op, transformed_args);
                op_trace = capture_inner.end_graph_capture();
            }  // end of inner graph capture

        }  // end of outer graph capture

        // extract memory footprint from the trace
        auto interleaved_storage_cores = device->num_banks(tt::tt_metal::BufferType::L1);
        size_t cb_peak_size_per_core = extract_circular_buffers_peak_size_per_core(op_trace);
        size_t l1_buffers_peak_per_core =
            extract_l1_buffer_allocation_peak_size_per_core(op_trace, interleaved_storage_cores);
        size_t l1_output_buffer_per_core =
            extract_l1_output_buffer_allocation_size_per_core(op_trace, interleaved_storage_cores);

        return QueryResponse{
            ExecutionStatus::Success, {cb_peak_size_per_core, l1_buffers_peak_per_core, l1_output_buffer_per_core}};

    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "op_constraints - error: {}", e.what());
        return QueryResponse{ExecutionStatus::Error, {0, 0, 0}, e.what()};
    }
}

}  // namespace ttnn::graph
