// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

enum class ExecutionStatus { Success, Error };

uint32_t extract_peak_L1_memory_usage(const nlohmann::json& trace);
uint32_t extract_l1_output_buffer_allocation_size_per_core(
    const ttnn::Tensor& output_tensor, size_t interleaved_storage_cores);
uint32_t extract_l1_buffer_allocation_peak_size_per_core(const nlohmann::json& trace, size_t interleaved_storage_cores);
uint32_t extract_circular_buffers_peak_size_per_core(const nlohmann::json& trace);

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

}  // namespace ttnn::graph
