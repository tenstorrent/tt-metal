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
// 4. Shards that the tensor topology labels as replicas of each other (mesh coordinates that differ only along
//    Replicate axes) are written once, and every record in the group points at that copy. The label is checked
//    against the data first: replicas must have the same size and, unless `ttnn::CONFIG`'s
//    `verify_replicated_shards_on_dump` is false, the same values (padding bytes are not compared). Every coordinate
//    the label lists must hold a shard (or be remote to this host), and every populated local shard must be covered by
//    the label. A tensor that fails any of these checks throws before the output file is created, so a rejected dump
//    leaves no file behind. The checks are `verify_shards_match_topology` (ttnn/tensor/topology_check.hpp); a caller
//    that keeps one copy per replica group itself can run them on a host tensor. In DISTRIBUTED_GATHER mode every rank
//    runs these checks once on the gathered tensor (every rank holds every shard after the gather; rank 0 does not
//    repeat them when it writes), so a rejected tensor throws on all ranks; a failed write on rank 0 still leaves the
//    other ranks in the final barrier. Known gap:
//    `DistributedHostBuffer::transform` rebuilds its container with every unpopulated shard marked remote, so after a
//    host-side layout or dtype conversion a partial tensor's missing local coordinates read as remote and pass the
//    exemption above. That is a missed rejection, never a false one.
void dump_tensor_flatbuffer(
    const std::string& file_name, const Tensor& tensor, DumpTensorMode mode = DumpTensorMode::DISTRIBUTED_GATHER);
Tensor load_tensor_flatbuffer(const std::string& file_name, tt::tt_metal::distributed::MeshDevice* device = nullptr);

}  // namespace ttnn
