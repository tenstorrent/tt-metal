// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/json/json.hpp"

#include <vector>

namespace ttnn::graph {

uint32_t extract_peak_memory_usage(const nlohmann::json& trace);

// Returns count of intermediate and output tensors
std::pair<int, int> count_intermediate_and_output_tensors(const nlohmann::json& trace);

std::vector<std::string> extract_calltrace(const nlohmann::json& trace);

size_t extract_output_L1_size(const nlohmann::json& trace);

} // namespace ttnn::graph
