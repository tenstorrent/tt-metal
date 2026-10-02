// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <fmt/format.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/cleanup.hpp>

#include "tensor/tensor_spec.hpp"
#include "tensor/flatbuffer/tensor_file_layout.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "ttnn/distributed/host_ccl.hpp"

namespace ttnn {
using tt::tt_metal::MemoryPin;

namespace {

// Distinguishes temporary files written by different threads of one process.
uint64_t next_temp_file_id() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

void dump_tensor_flatbuffer_impl(const std::string& file_name, const Tensor& tensor, DumpTensorMode mode) {
    Tensor cpu_tensor = tensor.cpu();

    if (mode == DumpTensorMode::DISTRIBUTED_GATHER) {
        // Dump tensor to disk from (global) rank 0 host.
        // Note we use global context as opposed to context embedded to the host-side tensor, since the tensor may
        // already be fully host-local. In this latter case, host buffer context will consist of a single (local) host
        // rank, and each host will attempt to flush the serialized tensor file to disk.
        cpu_tensor = ttnn::distributed::host_ccl::all_gather(cpu_tensor);
        const auto& ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        if (ctx->rank() != tt::tt_metal::distributed::multihost::Rank(0)) {
            ctx->barrier();
            return;
        }
    }

    // Write to a private temporary sibling and rename it into place. rename(2) is atomic, so a
    // concurrent reader (another process sharing the same tensor cache, e.g. the engines of a
    // multi-process data-parallel vLLM server) sees either no file or a complete one, never a
    // half-written file that load_tensor_flatbuffer would accept and then crash on. Two writers
    // racing on the same name each produce a complete file; the last rename wins.
    //
    // The temporary file is created exclusively: containers sharing one cache can share a pid, so
    // the name alone is not unique. It takes the destination's permissions when a destination
    // exists, so replacing a restricted cache file does not widen access. A writer killed outright
    // leaves its temporary file behind; the name does not end in .tensorbin, so cache listings
    // that glob for tensorbins never pick it up.
    mode_t file_mode = 0666;
    struct stat destination_stat{};
    if (stat(file_name.c_str(), &destination_stat) == 0) {
        file_mode = destination_stat.st_mode & 07777;
    }
    std::string temp_file_name;
    int fd = -1;
    do {
        temp_file_name = fmt::format("{}.tmp.{}.{}", file_name, getpid(), next_temp_file_id());
        fd = open(temp_file_name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, file_mode);
    } while (fd == -1 && errno == EEXIST);
    TT_FATAL(fd != -1, "Cannot create \"{}\": errno={} \"{}\"", temp_file_name, errno, strerror(errno));
    FILE* output_file = nullptr;
    bool renamed = false;
    auto cleanup = ttsl::make_cleanup([&output_file, &fd, &temp_file_name, &renamed]() {
        if (output_file != nullptr) {
            if (fclose(output_file) != 0) {
                log_warning(tt::LogAlways, "Failed to close \"{}\"", temp_file_name);
            }
        } else if (fd != -1) {
            close(fd);
        }
        if (!renamed) {
            unlink(temp_file_name.c_str());
        }
    });
    output_file = fdopen(fd, "wb");
    TT_FATAL(
        output_file != nullptr,
        "Cannot open \"{}\" for writing: errno={} \"{}\"",
        temp_file_name,
        errno,
        strerror(errno));

    std::vector<SerializedTensorBuffer> buffers;
    flatbuffers::FlatBufferBuilder builder;
    auto tensor_offset = ttnn::to_flatbuffer(cpu_tensor, builder, buffers);
    builder.Finish(tensor_offset);

    write_tensor_file(output_file, temp_file_name, builder, buffers);

    TT_FATAL(fflush(output_file) == 0, "Cannot flush \"{}\": errno={} \"{}\"", temp_file_name, errno, strerror(errno));
    // Close before publishing: a deferred write error (ENOSPC, a network file system) surfaces at
    // fclose, and only a fully written file may be renamed into place.
    const int close_rc = fclose(output_file);
    output_file = nullptr;
    fd = -1;
    TT_FATAL(close_rc == 0, "Cannot close \"{}\": errno={} \"{}\"", temp_file_name, errno, strerror(errno));
    TT_FATAL(
        rename(temp_file_name.c_str(), file_name.c_str()) == 0,
        "Cannot rename \"{}\" to \"{}\": errno={} \"{}\"",
        temp_file_name,
        file_name,
        errno,
        strerror(errno));
    renamed = true;

    if (mode == DumpTensorMode::DISTRIBUTED_GATHER) {
        const auto& ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        ctx->barrier();
    }
}

}  // namespace

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor, DumpTensorMode mode) {
    dump_tensor_flatbuffer_impl(file_name, tensor, mode);
}

Tensor load_tensor_flatbuffer(const std::string& file_name, tt::tt_metal::distributed::MeshDevice* device) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    size_t file_size = file_stat.st_size;
    TT_FATAL(file_size >= sizeof(uint64_t), "Tensor file \"{}\" is too small to be valid", file_name);

    // Mmap the file to read tensor data lazily.
    std::shared_ptr<void> mapping = map_tensor_file(fd, file_size, file_name);
    MemoryPin memory_pin(mapping);

    auto* file_data = static_cast<std::byte*>(mapping.get());
    uint64_t header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));
    TT_FATAL(
        sizeof(header_size) + header_size <= file_size,
        "Tensor file \"{}\" is truncated or corrupt (header_size={}, file_size={})",
        file_name,
        header_size,
        file_size);

    const auto* header_start = reinterpret_cast<const std::uint8_t*>(file_data) + sizeof(header_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header size is too large; this most likely indicates data corruption.");
    flatbuffers::Verifier verifier(header_start, header_size);
    TT_FATAL(
        ttnn::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor data; this most likely indicates data corruption.");
    const auto* fb_tensor = ttnn::flatbuffer::GetTensor(header_start);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = file_size - data_offset;

    std::byte* data_region = file_data + data_offset;
    TT_FATAL(
        (reinterpret_cast<uintptr_t>(data_region) & (kMinTensorDataAlignment - 1)) == 0,
        "Tensor data pointer must be {}-byte aligned!",
        kMinTensorDataAlignment);

    Tensor tensor = ttnn::from_flatbuffer(fb_tensor, ttsl::Span<std::byte>(data_region, data_size), memory_pin);
    if (device != nullptr) {
        tensor = tensor.to_device(device, tensor.tensor_spec().memory_config());
    }
    return tensor;
}

}  // namespace ttnn
