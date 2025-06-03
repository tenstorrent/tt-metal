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

Tensor load_tensor(const std::string& file_name, distributed::MeshDevice* device = nullptr);

void dump_memory_config(FILE* output_file, const MemoryConfig& memory_config);
void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config);

MemoryConfig load_memory_config(FILE* input_file);
MemoryConfig load_memory_config(const std::string& file_name);

// Functions to load and dump tensor to file using FlatBuffer format with inline file storage.
// Only inline file storage (data stored in same file) is currently supported:
// 1. Tensor metadata is serialized and stored as file "header", while the rest of the file is used as a data region for
//    tensor data.
// 2. Metadata includes data offsets and sizes for tensor / tensor shards (multi device context).
// TODO: #22259 - the format is not yet finalized, and is not stable. Avoid using it in production.
void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor);
Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device = nullptr);

}  // namespace tt::tt_metal
