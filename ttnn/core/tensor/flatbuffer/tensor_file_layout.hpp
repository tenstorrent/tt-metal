// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <tt-metalium/host_buffer.hpp>

namespace ttnn {

// Serialized tensor files are laid out as:
//
//   [0, 8)                      uint64 header_size
//   [8, 8 + header_size)        flatbuffer header, zero-padded up to `kTensorDataAlignment`
//   [8 + header_size, EOF)      shard buffers, each starting on `kTensorDataAlignment`
//
// Because the padding is folded into `header_size` and every shard records its own offset in the header,
// readers derive all positions from the file itself and need no knowledge of the alignment. Files written
// before this alignment existed therefore keep loading unchanged.

// Alignment of the tensor data region and of every shard buffer within it. A reader that `mmap`s the file at
// offset 0 therefore gets shard pointers it can hand to the driver as pinned memory without copying first.
// 64 is Blackhole's PCIe transfer alignment; Wormhole's is 32. This is deliberately a fixed number rather than
// the running architecture's alignment, because a file written on one architecture has to stay usable on the
// other, so the format has to commit to the largest.
inline constexpr uint64_t kTensorDataAlignment = 64;

// What a reader may assume of the data region. Files predating `kTensorDataAlignment` are only aligned to the
// 8 bytes that `header_size` itself forces, so this, not `kTensorDataAlignment`, is the invariant to check on
// load -- checking the stronger one would reject every file written before it existed.
inline constexpr uint64_t kMinTensorDataAlignment = alignof(std::uint64_t);

// A shard buffer paired with the byte offset, relative to the start of the tensor data region, at which it
// must be written. Offsets are aligned to `kTensorDataAlignment`, so consecutive buffers are generally not
// adjacent; the gaps are zero-filled on write.
struct SerializedTensorBuffer {
    tt::tt_metal::HostBuffer buffer;
    uint64_t offset = 0;
};

// Writes a finished flatbuffer header and its shard buffers to `file` in the layout described above.
// `file_name` is used for error reporting only.
void write_tensor_file(
    FILE* file,
    const std::string& file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    const std::vector<SerializedTensorBuffer>& buffers);

}  // namespace ttnn
