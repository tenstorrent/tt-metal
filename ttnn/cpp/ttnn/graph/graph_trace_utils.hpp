// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/json/json.hpp"

#include <vector>

namespace ttnn::graph {

uint32_t extract_peak_memory_usage(const nlohmann::json& trace);

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace);

std::vector<std::string> extract_calltrace(const nlohmann::json& trace);

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace);

struct OutputSizes {
    size_t total_L1_size = 0;
    size_t total_DRAM_size = 0;

    std::vector<uint32_t> L1_sizes;
    std::vector<uint32_t> DRAM_sizes;
};
OutputSizes extract_output_sizes(const nlohmann::json& trace);

std::vector<ttnn::Shape> extract_output_shapes(const nlohmann::json& trace);

} // namespace ttnn::graph
