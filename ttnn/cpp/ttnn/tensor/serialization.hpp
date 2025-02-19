// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <string>
#include <unordered_map>

namespace tt::tt_metal {

void dump_tensor(
    const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy);

Tensor load_tensor(const std::string& file_name, IDevice* device = nullptr);
Tensor load_tensor(const std::string& file_name, distributed::MeshDevice* device = nullptr);

void dump_memory_config(FILE* output_file, const MemoryConfig& memory_config);
void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config);

MemoryConfig load_memory_config(FILE* input_file);
MemoryConfig load_memory_config(const std::string& file_name);

}  // namespace tt::tt_metal
