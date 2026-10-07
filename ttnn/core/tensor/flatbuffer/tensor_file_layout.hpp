// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

// A shard buffer paired with the byte offset, relative to the start of the tensor data region, at which it
// must be written. Offsets are aligned to `kTensorDataAlignment`, so consecutive buffers are generally not
// adjacent; the gaps are zero-filled on write.
struct SerializedTensorBuffer {
    tt::tt_metal::HostBuffer buffer;
    uint64_t offset = 0;
};

// Writes a finished flatbuffer header and its shard buffers to `file_name` in the layout described above.
//
// The data goes to a new file in the same directory, which is then renamed over `file_name`, so an existing file is
// replaced rather than rewritten in place. rename(2) is atomic: a concurrent reader, such as another process sharing
// the same tensor cache, sees either the previous file or the complete new one, never a half-written file, and of two
// writers racing on one name the last rename wins. A tensor loaded from the old file keeps it mapped and may have it
// pinned for device uploads (see `map_tensor_file`). Rewriting that file in place would change the loaded tensor's
// contents under it, or kill its readers with SIGBUS where the new file is shorter, while uploads from a cached pin
// kept sending the old pages. Replacing it leaves the old file intact until its last mapping goes away. When
// `file_name` is a symlink, the file it points to is replaced, not the link.
void write_tensor_file(
    const std::string& file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    ttsl::Span<const SerializedTensorBuffer> buffers);

// Maps all `file_size` bytes of the tensor file open on the read-only descriptor `fd` with PROT_READ, for the readers
// named above. The returned owner unmaps the file when the last reference is released, so the caller can hand it to
// a `MemoryPin`. `file_name` is used for error reporting only.
//
// The mapping is MAP_SHARED wherever the filesystem allows it, because uploads pin it read-only as a device DMA
// source. A long-term pin of a MAP_PRIVATE file mapping makes the kernel first copy every page into private anonymous
// memory (copy-on-write unshare), which is slower than the upload itself and doubles resident memory; a shared
// mapping is pinned in place. The cost of pinning in place is that the device reads the file's page cache: writing
// the file in place while a pin is cached changes what later uploads from that pin transfer, where the private copies
// made for a MAP_PRIVATE pin would not have changed. `write_tensor_file` replaces a file instead of writing it in
// place, so only other writers can do that.
//
// Some filesystems accept MAP_PRIVATE but refuse MAP_SHARED: FUSE in direct-I/O mode fails a shared mapping with
// ENODEV unless the server allows it. The mapping then falls back to MAP_PRIVATE, which loads correctly and pays the
// copy above on a pinned upload.
std::shared_ptr<void> map_tensor_file(int fd, size_t file_size, std::string_view file_name);

}  // namespace ttnn
