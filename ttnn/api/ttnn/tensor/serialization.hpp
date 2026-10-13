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
// 3. The data region and every shard within it start on a 64-byte boundary, so a caller that `mmap`s a file
//    written by `dump_tensor_flatbuffer` can use the shard pointers as a pinned DMA source without copying.
//    `load_tensor_flatbuffer` checks only that the data region itself is 8-byte aligned; shards in files written
//    before this layout can be aligned more weakly, so a caller that depends on the 64-byte guarantee has to check
//    the pointers it gets.
void dump_tensor_flatbuffer(
    const std::string& file_name, const Tensor& tensor, DumpTensorMode mode = DumpTensorMode::DISTRIBUTED_GATHER);
Tensor load_tensor_flatbuffer(const std::string& file_name, tt::tt_metal::distributed::MeshDevice* device = nullptr);

// Experimental. Loads a tensor file onto `device` like `load_tensor_flatbuffer`, but streams the shards from disk with
// O_DIRECT reads into a small set of staging buffers instead of mapping the file. The page cache is bypassed in both
// directions: an already-cached file is still read from disk, and loading does not fill the cache.
// Falls back to `load_tensor_flatbuffer` when the shards do not cover the whole mesh, when the memory config is not
// interleaved, or when the file system rejects O_DIRECT.
Tensor load_tensor_flatbuffer_direct(const std::string& file_name, tt::tt_metal::distributed::MeshDevice& device);

}  // namespace ttnn
