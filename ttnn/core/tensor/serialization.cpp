// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/cleanup.hpp>

#include "tensor/tensor_spec.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "ttnn/distributed/host_ccl.hpp"

namespace tt::tt_metal {
namespace {

void safe_fwrite(const void* buffer, size_t size, size_t count, FILE* file) {
    TT_FATAL(fwrite(buffer, size, count, file) == count, "Failed to write tensor data: file write failed");
}

constexpr std::uint32_t kFlatbufferAlignment = alignof(std::uint64_t);

}  // namespace

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor) {
    Tensor cpu_tensor = tensor.cpu();

    // Dump tensor to disk from rank 0 host.
    cpu_tensor = ttnn::distributed::host_ccl::all_gather(cpu_tensor);
    const auto& ctx = cpu_tensor.host_storage().buffer().context();
    if (ctx->rank() == tt::tt_metal::distributed::multihost::Rank(0)) {
        FILE* output_file = fopen(file_name.c_str(), "wb");
        TT_FATAL(output_file != nullptr, "Cannot open \"{}\"", file_name);
        auto cleanup = ttsl::make_cleanup([f = output_file]() {
            if (f && fclose(f) != 0) {
                log_warning(tt::LogAlways, "Failed to close file");
            }
        });

        std::vector<HostBuffer> buffers;
        flatbuffers::FlatBufferBuilder builder;
        auto tensor_offset = ttnn::to_flatbuffer(cpu_tensor, builder, buffers);
        // To be able to read flatbuffer data with `mmap` safely, make sure the serialized flatbuffer is aligned to at
        // least 8 bytes, just like `header_size`. Individual `buffers` are aligned according to their element size,
        // which is already what we need for `mmap` to work.
        builder.Align(kFlatbufferAlignment);
        builder.Finish(tensor_offset);

        uint64_t header_size = builder.GetSize();
        safe_fwrite(&header_size, sizeof(header_size), 1, output_file);
        safe_fwrite(builder.GetBufferPointer(), header_size, 1, output_file);

        for (const auto& buffer : buffers) {
            auto buffer_view = buffer.view_bytes();
            safe_fwrite(buffer_view.data(), buffer_view.size(), 1, output_file);
        }
    }
    ctx->barrier();
}

Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\"", file_name);
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    size_t file_size = file_stat.st_size;

    // Mmap the file to read tensor data lazily.
    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TT_FATAL(mmap_addr != MAP_FAILED, "Failed to mmap file \"{}\": {}", file_name, strerror(errno));

    std::shared_ptr<void> mmap_ptr(mmap_addr, [file_size](void* addr) { munmap(addr, file_size); });
    MemoryPin memory_pin(mmap_ptr);

    auto* file_data = static_cast<std::byte*>(mmap_addr);
    uint64_t header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));

    const auto* header_start = reinterpret_cast<const std::uint8_t*>(file_data) + sizeof(header_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header size is too large; this most likely indicates data corruption.");
    flatbuffers::Verifier verifier(header_start, header_size);
    TT_FATAL(
        ttnn::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor data; this most likely indicates data corruption.");
    auto fb_tensor = ttnn::flatbuffer::GetTensor(header_start);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = file_size - data_offset;

    std::byte* data_region = file_data + data_offset;
    TT_FATAL(
        (reinterpret_cast<uintptr_t>(data_region) & (kFlatbufferAlignment - 1)) == 0,
        "Tensor data pointer must be 8-byte aligned!");

    Tensor tensor = ttnn::from_flatbuffer(fb_tensor, tt::stl::Span<std::byte>(data_region, data_size), memory_pin);
    if (device != nullptr) {
        tensor = tensor.to_device(device, tensor.tensor_spec().memory_config());
    }
    return tensor;
}

}  // namespace tt::tt_metal
