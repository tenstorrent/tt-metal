// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal {

enum class DumpTensorMode : std::uint8_t {
    DISTRIBUTED_GATHER = 0,
    LOCAL = 1,
};

// Functions to load and dump tensor to file using FlatBuffer format with inline file storage.
// Only inline file storage (data stored in same file) is currently supported:
// 1. Tensor metadata is serialized and stored as file "header", while the rest of the file is used as a data region for
//    tensor data.
// 2. Metadata includes data offsets and sizes for tensor / tensor shards (multi device context).
void dump_tensor_flatbuffer(
    const std::string& file_name, const Tensor& tensor, DumpTensorMode mode = DumpTensorMode::DISTRIBUTED_GATHER);
Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device = nullptr);

// Coalesces distributed BFLOAT4_B tensorbins into one host tensor concatenated along logical dimension 0.
// Source payloads are copied shard-by-shard without materializing the individual input tensors.
Tensor coalesce_tensorbins(const std::vector<std::string>& input_file_names);

// Reinterprets the base of a coalesced device tensor using a compatible single-input host tensor spec.
// The returned tensor shares ownership of the packed device allocation.
Tensor alias_coalesced_tensor(const Tensor& packed_device_tensor, const Tensor& template_host_tensor);

}  // namespace tt::tt_metal
