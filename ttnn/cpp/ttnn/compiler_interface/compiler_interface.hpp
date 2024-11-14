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

QueryResponse extract_data_from_trace(const nlohmann::json &trace, size_t interleaved_storage_cores) {
    size_t cb_peak_size_per_core = graph::extract_circular_buffers_peak_size_per_core(trace);
    size_t l1_buffers_peak_per_core =
        graph::extract_l1_buffer_allocation_peak_size_per_core(trace, interleaved_storage_cores);
    size_t l1_output_buffer_per_core =
        graph::extract_l1_output_buffer_allocation_size_per_core(trace, interleaved_storage_cores);
    bool constraint_valid = true;

    return QueryResponse{
        ExecutionStatus::Success, {cb_peak_size_per_core, l1_buffers_peak_per_core, l1_output_buffer_per_core}};
}

}  // namespace ttnn::compiler_interface
