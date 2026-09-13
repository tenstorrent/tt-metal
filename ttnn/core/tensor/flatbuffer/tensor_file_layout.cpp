// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_file_layout.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <string_view>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn {
namespace {

void safe_fwrite_bytes(
    const void* buffer, size_t bytes, FILE* file, const std::string& filename, std::string_view what) {
    TT_FATAL(bytes > 0, "Expected to write > 0 bytes to file");

    // Use byte-wise fwrite so we can detect partial writes
    const size_t written = fwrite(buffer, /*size=*/1, /*count=*/bytes, file);
    TT_FATAL(
        written == bytes,
        "Failed to write {} to \"{}\": wrote {}/{} bytes (ferror={}, errno={} \"{}\")",
        what,
        filename,
        written,
        bytes,
        ferror(file),
        errno,
        strerror(errno));
}

// Zero-fills `bytes` bytes. Aligning to `kTensorDataAlignment` never needs more than that many bytes of padding.
void write_padding(size_t bytes, FILE* file, const std::string& filename) {
    if (bytes == 0) {
        return;
    }
    static constexpr std::array<std::byte, kTensorDataAlignment> zeros{};
    TT_FATAL(bytes < zeros.size(), "Padding of {} bytes exceeds the {}-byte alignment", bytes, zeros.size());
    safe_fwrite_bytes(zeros.data(), bytes, file, filename, "padding");
}

}  // namespace

void write_tensor_file(
    FILE* file,
    const std::string& file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    const std::vector<SerializedTensorBuffer>& buffers) {
    // Pad the header so the data region starts on `kTensorDataAlignment`. The padding is counted in `header_size`,
    // so a reader that only knows about the 8-byte minimum still finds the data region where it expects it.
    const uint64_t flatbuffer_size = builder.GetSize();
    const uint64_t header_size = tt::align(sizeof(uint64_t) + flatbuffer_size, kTensorDataAlignment) - sizeof(uint64_t);

    safe_fwrite_bytes(&header_size, sizeof(header_size), file, file_name, "tensor header size");
    safe_fwrite_bytes(builder.GetBufferPointer(), flatbuffer_size, file, file_name, "tensor header");
    write_padding(header_size - flatbuffer_size, file, file_name);

    uint64_t bytes_written = 0;
    for (const auto& [buffer, offset] : buffers) {
        auto buffer_view = buffer.view_bytes();
        TT_FATAL(!buffer_view.empty(), "Unexpected empty buffer during tensor serialization");
        TT_FATAL(
            offset >= bytes_written,
            "Shard buffer offset {} overlaps the preceding buffer, which ends at {}",
            offset,
            bytes_written);
        write_padding(offset - bytes_written, file, file_name);
        safe_fwrite_bytes(buffer_view.data(), buffer_view.size(), file, file_name, "tensor data");
        bytes_written = offset + buffer_view.size();
    }

    TT_FATAL(fflush(file) == 0, "Failed to flush \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
}

}  // namespace ttnn
