// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include <flatbuffers/flatbuffers.h>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/cleanup.hpp>

#include "distributed/distributed_tensor_config.hpp"
#include "tensor/tensor_spec.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/types.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"

namespace tt::tt_metal {
namespace {

void validate_version(uint8_t version_id) {
    TT_FATAL(
        version_id >= 5,
        "Version {} is no longer supported. Please update your saved data to the supported version (5).",
        version_id);
    TT_FATAL(
        version_id <= VERSION_ID,
        "Version mismatch: the serialized tensor was created with version {} but is "
        "being loaded by a loader with version {}. Please update your saved data or your "
        "loader so that both versions match.",
        version_id,
        VERSION_ID);
}

auto make_file_closer(FILE* file) {
    return ttsl::make_cleanup([file]() {
        if (file) {
            if (fclose(file) != 0) {
                log_warning(tt::LogAlways, "Failed to close file");
            }
        }
    });
}

void safe_fread(void* buffer, size_t size, size_t count, FILE* file) {
    if (fread(buffer, size, count, file) != count) {
        TT_THROW("Failed to read tensor data, file must be corrupted");
    }
}

void safe_fwrite(const void* buffer, size_t size, size_t count, FILE* file) {
    if (fwrite(buffer, size, count, file) != count) {
        TT_THROW("Failed to write tensor data: file write failed");
    }
}

constexpr std::uint32_t kFlatbufferAlignment = alignof(std::uint64_t);

}  // namespace

void dump_memory_config(FILE* output_file, const MemoryConfig& memory_config) {
    safe_fwrite(&VERSION_ID, sizeof(VERSION_ID), 1, output_file);
    flatbuffers::FlatBufferBuilder builder;
    auto flat_config = ttnn::to_flatbuffer(memory_config, builder);
    builder.Finish(flat_config);
    uint64_t buf_size = builder.GetSize();
    safe_fwrite(&buf_size, sizeof(buf_size), 1, output_file);
    safe_fwrite(builder.GetBufferPointer(), buf_size, 1, output_file);
}

void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    if (not output_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    auto cleanup = make_file_closer(output_file);
    dump_memory_config(output_file, memory_config);
}

MemoryConfig load_memory_config(FILE* input_file) {
    std::uint8_t version_id;
    safe_fread(&version_id, sizeof(version_id), 1, input_file);
    validate_version(version_id);

    uint64_t bin_size = 0;
    safe_fread(&bin_size, sizeof(bin_size), 1, input_file);
    std::vector<uint8_t> bin(bin_size);
    safe_fread(bin.data(), bin_size, 1, input_file);
    flatbuffers::Verifier verifier(bin.data(), bin_size);
    if (!verifier.VerifyBuffer<ttnn::flatbuffer::MemoryConfig>()) {
        TT_THROW("MemoryConfig deserialization failed: invalid buffer");
    }
    auto mem_config = flatbuffers::GetRoot<ttnn::flatbuffer::MemoryConfig>(bin.data());
    return ttnn::from_flatbuffer(mem_config);
}

MemoryConfig load_memory_config(const std::string& file_name) {
    FILE* input_file = fopen(file_name.c_str(), "rb");
    if (not input_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    auto cleanup = make_file_closer(input_file);
    return load_memory_config(input_file);
}

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    TT_FATAL(output_file != nullptr, "Cannot open \"{}\"", file_name);
    auto cleanup = make_file_closer(output_file);

    Tensor cpu_tensor = tensor.cpu();

    std::vector<HostBuffer> buffers;
    flatbuffers::FlatBufferBuilder builder;
    auto tensor_offset = ttnn::to_flatbuffer(cpu_tensor, builder, buffers);
    // To be able to read flatbuffer data with `mmap` safely, make sure the serialized flatbuffer is aligned to at least
    // 8 bytes, just like `header_size`. Individual `buffers` are aligned according to their element size, which is
    // already what we need for `mmap` to work.
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
