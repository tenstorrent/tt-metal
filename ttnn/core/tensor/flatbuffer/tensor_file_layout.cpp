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

void safe_fwrite_bytes(const void* buffer, size_t bytes, FILE* file, std::string_view filename, std::string_view what) {
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
void write_padding(size_t bytes, FILE* file, std::string_view filename) {
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
    std::string_view file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    ttsl::Span<const SerializedTensorBuffer> buffers) {
    // Pad the header so the data region starts on `kTensorDataAlignment`. The padding is counted in `header_size`,
    // so a reader that only knows about the 8-byte minimum still finds the data region where it expects it.
    const uint64_t flatbuffer_size = builder.GetSize();
    const uint64_t header_size = tt::align(sizeof(uint64_t) + flatbuffer_size, kTensorDataAlignment) - sizeof(uint64_t);

    safe_fwrite_bytes(&header_size, sizeof(header_size), file, file_name, "tensor header size");
    safe_fwrite_bytes(builder.GetBufferPointer(), flatbuffer_size, file, file_name, "tensor header");
    write_padding(header_size - flatbuffer_size, file, file_name);

    // Offset, relative to the start of the data region, one past the last buffer written so far.
    uint64_t data_region_end = 0;
    for (const auto& [buffer, offset] : buffers) {
        auto buffer_view = buffer.view_bytes();
        TT_FATAL(!buffer_view.empty(), "Unexpected empty buffer during tensor serialization");
        TT_FATAL(
            offset >= data_region_end,
            "Shard buffer offset {} overlaps the preceding buffer, which ends at {}",
            offset,
            data_region_end);
        write_padding(offset - data_region_end, file, file_name);
        safe_fwrite_bytes(buffer_view.data(), buffer_view.size(), file, file_name, "tensor data");
        data_region_end = offset + buffer_view.size();
    }

    TT_FATAL(fflush(file) == 0, "Failed to flush \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
}

}  // namespace ttnn
