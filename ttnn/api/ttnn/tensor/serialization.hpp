// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace ttnn {

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
    const std::string& file_name, const ttnn::Tensor& tensor, DumpTensorMode mode = DumpTensorMode::DISTRIBUTED_GATHER);
Tensor load_tensor_flatbuffer(const std::string& file_name, tt::tt_metal::distributed::MeshDevice* device = nullptr);

}  // namespace ttnn

// Compatibility aliases - ttnn tensor infrastructure has moved to the ttnn namespace.
namespace tt::tt_metal {

using DumpTensorMode [[deprecated("use ttnn::DumpTensorMode instead. This alias may be removed after Jun 2026.")]] =
    ttnn::DumpTensorMode;

template <int = 0>
[[deprecated("use ttnn::dump_tensor_flatbuffer instead. This alias may be removed after Jun 2026.")]]
inline void dump_tensor_flatbuffer(
    const std::string& file_name,
    const ttnn::Tensor& tensor,
    ttnn::DumpTensorMode mode = ttnn::DumpTensorMode::DISTRIBUTED_GATHER) {
    ttnn::dump_tensor_flatbuffer(file_name, tensor, mode);
}

template <int = 0>
[[deprecated("use ttnn::load_tensor_flatbuffer instead. This alias may be removed after Jun 2026.")]]
inline ttnn::Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device = nullptr) {
    return ttnn::load_tensor_flatbuffer(file_name, device);
}

}  // namespace tt::tt_metal
