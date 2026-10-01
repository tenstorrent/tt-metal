// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>
#include <string_view>

#include <flatbuffers/flatbuffers.h>

#include <tt_stl/span.hpp>
#include <tt-metalium/host_buffer.hpp>

namespace ttnn {

// Serialized tensor files are laid out as:
//
//   [0, 8)                      uint64 header_size
//   [8, 8 + header_size)        flatbuffer header, zero-padded up to `kTensorDataAlignment`
//   [8 + header_size, EOF)      shard buffers, each starting on `kTensorDataAlignment`
//
// The flatbuffer header is a `ttnn.flatbuffer.Tensor` (tensor.fbs). From schema version 1 on it opens with the root
// offset followed by the file identifier "TTNB" (file bytes [12, 16)) and records `schema_version`; see
// `kTensorFileSchemaVersion` for what readers do with both.
//
// Because the padding is folded into `header_size` and every shard records its own offset in the header, a reader
// derives every position from the file itself. The alignment is a writer-side choice that the format does not
// encode, so a file whose data region is aligned to at least `kMinTensorDataAlignment` loads correctly.
//
// `write_tensor_file` below is the only writer; the readers live in `ttnn/core/tensor/serialization.cpp` and
// `ttnn/core/tensor/overlapped_serialization.cpp`.

// Alignment of the tensor data region and of every shard buffer within it, so that a reader which `mmap`s the file
// gets shard pointers a device transfer can use as a pinned DMA source directly, instead of paying for a staging
// copy or for sending the unaligned head inline.
// The value is the largest PCIe transfer alignment across architectures: `PCIE_ALIGNMENT` in `noc_parameters.h` is
// 64 on Blackhole and Quasar and 32 on Wormhole, and `tt::tt_metal::hal::get_pcie_alignment()` is its runtime
// accessor. The format commits to a fixed number rather than the running architecture's, because a file written on
// one architecture has to stay usable on the others.
inline constexpr uint64_t kTensorDataAlignment = 64;

// The weakest alignment a data region can have: `header_size` follows an 8-byte field and is itself a count of
// bytes, so 8 is structurally forced and nothing beyond it is. Files on disk may sit anywhere at or above this, so
// this, not `kTensorDataAlignment`, is what a load-time check can require.
inline constexpr uint64_t kMinTensorDataAlignment = alignof(std::uint64_t);

// Revision of the header schema (tensor.fbs) that this writer records in `Tensor.schema_version`:
//   0  pre-versioning. The field and the file identifier are absent. `tensor_topology` may be absent (files written
//      before that field existed load fully replicated) and, when present, was written without being checked
//      against the shards.
//   1  the file identifier is present, `tensor_topology` is always recorded, and the writer checked it against the
//      shards: replica groups hold identical bytes, every labelled coordinate is backed by a local or remote shard,
//      and every populated shard is covered by the label.
// A reader accepts every version up to its own and rejects newer ones, so bump this whenever the writer starts
// recording something a reader has to understand, or guaranteeing something a reader may rely on.
inline constexpr uint32_t kTensorFileSchemaVersion = 1;

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
    std::string_view file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    ttsl::Span<const SerializedTensorBuffer> buffers);

}  // namespace ttnn
