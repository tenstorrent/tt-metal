// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/overlapped_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <tt_stl/cleanup.hpp>

#include "tensor/flatbuffer/overlapped_tensor_flatbuffer.hpp"
#include "ttnn/distributed/host_ccl.hpp"

namespace tt::tt_metal {
namespace {

void safe_fwrite(const void* buffer, size_t bytes, FILE* file, const std::string& filename, std::string_view what) {
    TT_FATAL(bytes > 0, "Expected to write > 0 bytes to file");
    const size_t written = fwrite(buffer, 1, bytes, file);
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

constexpr std::uint32_t kOverlappedFlatbufferAlignment = alignof(std::uint64_t);

}  // namespace

void dump_overlapped_tensors(const std::string& file_name, const std::vector<OverlappedTensorView>& views) {
    TT_FATAL(!views.empty(), "Need at least one view to serialize");

    const auto& fused_ref = views[0].fused_tensor;
    for (size_t i = 1; i < views.size(); ++i) {
        TT_FATAL(
            views[i].fused_tensor.tensor_id == fused_ref.tensor_id,
            "All OverlappedTensorViews must reference the same fused tensor (view {} differs)",
            i);
    }

    Tensor cpu_tensor = views[0].fused_tensor.cpu();
    cpu_tensor = ttnn::distributed::host_ccl::all_gather(cpu_tensor);

    // Build a temporary view list with the CPU-side fused tensor for serialization.
    std::vector<OverlappedTensorView> cpu_views = views;
    for (auto& v : cpu_views) {
        v.fused_tensor = cpu_tensor;
    }

    const auto& ctx = distributed::multihost::DistributedContext::get_current_world();
    if (ctx->rank() == distributed::multihost::Rank(0)) {
        FILE* output_file = fopen(file_name.c_str(), "wb");
        TT_FATAL(
            output_file != nullptr,
            "Cannot open \"{}\" for writing: errno={} \"{}\"",
            file_name,
            errno,
            strerror(errno));
        auto cleanup = ttsl::make_cleanup([f = output_file, &file_name]() {
            if (f && fclose(f) != 0) {
                log_warning(tt::LogAlways, "Failed to close \"{}\"", file_name);
            }
        });

        std::vector<HostBuffer> buffers;
        flatbuffers::FlatBufferBuilder builder;
        auto root_offset = ttnn::overlapped_tensors_to_flatbuffer(cpu_views, builder, buffers);
        builder.Align(kOverlappedFlatbufferAlignment);
        builder.Finish(root_offset);

        const uint64_t header_size = builder.GetSize();
        safe_fwrite(&header_size, sizeof(header_size), output_file, file_name, "header size");
        safe_fwrite(builder.GetBufferPointer(), header_size, output_file, file_name, "header");

        for (const auto& buffer : buffers) {
            auto buffer_view = buffer.view_bytes();
            TT_FATAL(!buffer_view.empty(), "Unexpected empty buffer during serialization");
            safe_fwrite(buffer_view.data(), buffer_view.size(), output_file, file_name, "tensor data");
        }

        TT_FATAL(
            fflush(output_file) == 0, "Failed to flush \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    }
    ctx->barrier();
}

std::vector<OverlappedTensorView> load_overlapped_tensors(
    const std::string& file_name, distributed::MeshDevice* device) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    size_t file_size = file_stat.st_size;
    TT_FATAL(file_size >= sizeof(uint64_t), "File \"{}\" is too small to be valid", file_name);

    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TT_FATAL(mmap_addr != MAP_FAILED, "Failed to mmap file \"{}\": {}", file_name, strerror(errno));

    std::shared_ptr<void> mmap_ptr(mmap_addr, [file_size](void* addr) { munmap(addr, file_size); });
    MemoryPin memory_pin(mmap_ptr);

    auto* file_data = static_cast<std::byte*>(mmap_addr);
    uint64_t header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));
    TT_FATAL(
        sizeof(header_size) + header_size <= file_size,
        "File \"{}\" is truncated or corrupt (header_size={}, file_size={})",
        file_name,
        header_size,
        file_size);

    const auto* header_start = reinterpret_cast<const std::uint8_t*>(file_data) + sizeof(header_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Header size is too large; this most likely indicates data corruption.");
    flatbuffers::Verifier verifier(header_start, header_size);
    TT_FATAL(
        ttnn::flatbuffer::VerifyOverlappedTensorsBuffer(verifier),
        "Cannot validate overlapped tensor data; this most likely indicates data corruption.");
    const auto* fb_root = ttnn::flatbuffer::GetOverlappedTensors(header_start);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = file_size - data_offset;

    std::byte* data_region = file_data + data_offset;
    TT_FATAL(
        (reinterpret_cast<uintptr_t>(data_region) & (kOverlappedFlatbufferAlignment - 1)) == 0,
        "Tensor data pointer must be 8-byte aligned!");

    auto views =
        ttnn::overlapped_tensors_from_flatbuffer(fb_root, tt::stl::Span<std::byte>(data_region, data_size), memory_pin);

    if (device != nullptr) {
        auto& fused = views[0].fused_tensor;
        fused = fused.to_device(device, fused.tensor_spec().memory_config());
        for (size_t i = 1; i < views.size(); ++i) {
            views[i].fused_tensor = fused;
        }
    }

    return views;
}

}  // namespace tt::tt_metal
