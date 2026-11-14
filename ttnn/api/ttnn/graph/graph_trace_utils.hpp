// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <vector>

namespace ttnn::graph {

struct OperationInfo {
    std::string operation_name;
    std::vector<std::string> arguments;
};

struct PeakMemoryUsagePerCore {
    /** circular buffers only; a Metal 2.0 program has none, so this is legitimately 0 for a ported op */
    size_t peak_cb = 0;
    size_t peak_l1 = 0;
    /** Metal 2.0 program-scope L1, reported per kind rather than folded into peak_cb */
    size_t peak_dataflow_buffer = 0;
    size_t peak_scratchpad = 0;
    /** every kind above, so a "does this fit in L1" check needs this one number */
    size_t peak_total = 0;
};

struct DRAMUsage {
    /** peak memory usage in bytes between begin_graph_capture and end_graph_capture */
    long long peak = 0;
    /** total memory allocated in bytes between begin_graph_capture and end_graph_capture */
    long long total_allocations = 0;
    /** total memory deallocated in bytes between begin_graph_capture and end_graph_capture */
    long long total_deallocations = 0;
};

enum class ExecutionStatus { Success, Error };

uint32_t extract_peak_L1_memory_usage(const nlohmann::json& trace);

// Returns the actual per-bank L1 allocation size for the output buffer. Throws for DRAM buffers.
uint32_t extract_l1_output_buffer_allocation_size_per_core(const ttnn::Tensor& output_tensor);

PeakMemoryUsagePerCore extract_resource_usage_per_core(const nlohmann::json& trace);

DRAMUsage extract_dram_usage(const nlohmann::json& trace);

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace);

std::vector<std::string> extract_calltrace(const nlohmann::json& trace);

std::vector<OperationInfo> extract_arguments(const nlohmann::json& trace);

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace);

struct TensorInfo {
    ttnn::Shape shape;
    uint32_t size = 0;
    tt::tt_metal::BufferType type = tt::tt_metal::BufferType::DRAM;

    bool operator==(const TensorInfo& other) const = default;
};

std::vector<TensorInfo> extract_output_info(const nlohmann::json& trace);

nlohmann::json extract_levelized_graph(const nlohmann::json& trace, size_t max_level = 1);

}  // namespace ttnn::graph
