// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "third_party/json/json.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttnn::compiler_interface {

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
 * @brief Extracts resource usage data from a given trace.
 *
 * @param trace The graph trace containing the graph operations data.
 * @param interleaved_storage_cores The number of interleaved storage cores used in the computation.
 * @return QueryResponse containing the execution status and resource usage data.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
QueryResponse extract_data_from_trace(const nlohmann::json& trace, size_t interleaved_storage_cores) {
    size_t cb_peak_size_per_core = graph::extract_circular_buffers_peak_size_per_core(trace);
    size_t l1_buffers_peak_per_core =
        graph::extract_l1_buffer_allocation_peak_size_per_core(trace, interleaved_storage_cores);
    size_t l1_output_buffer_per_core =
        graph::extract_l1_output_buffer_allocation_size_per_core(trace, interleaved_storage_cores);
    bool constraint_valid = true;

    return QueryResponse{
        ExecutionStatus::Success, {cb_peak_size_per_core, l1_buffers_peak_per_core, l1_output_buffer_per_core}};
}

/**
 * @brief Captures the graph operations and extracts resource usage constraints.
 *
 * This function captures the graph operations by invoking the provided callable,
 * then extracts and returns the resource usage constraints from the captured trace.
 *
 * @tparam Callable The type of the callable object that will be invoked to capture the graph operations.
 * @param device A pointer to the Device object, which provides information about the compute grid size.
 * @param callable The callable object that will be invoked to capture the graph operations. It must run the op and
 * return a valid graph trace.
 * @return QueryResponse containing the execution status and resource usage constraints.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
template <class Callable>
QueryResponse op_constraints(Device* device, Callable&& callable) {
    try {
        ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
        const nlohmann::json op_trace = callable();
        ttnn::graph::GraphProcessor::end_graph_capture();

        auto interleaved_storage_cores =
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
        return extract_data_from_trace(op_trace, interleaved_storage_cores);
    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "compiler_interface - error: {}", e.what());
        return QueryResponse{ExecutionStatus::Error, {0, 0, 0}, e.what()};
    }
}

}  // namespace ttnn::compiler_interface
