// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <flatbuffers/verifier.h>

#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/cleanup.hpp>

#include "tensor/flatbuffer/tensor_flatbuffer.hpp"

namespace ttnn {
namespace {

using tt::tt_metal::distributed::MeshBuffer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::ShardDataTransfer;

// O_DIRECT needs the file offset, length and buffer address aligned to the device's logical block size. 4 KiB covers
// every block size in practice; a stricter value only widens each read slightly.
constexpr uint64_t kDirectIoAlignment = 4096;
constexpr uint64_t kTargetChunkBytes = 8ull * 1024 * 1024;
// Chunks are interleaved across shards, so one batch spans up to this many devices and the dispatch thread pool
// copies them in parallel; a single copy thread is slower than the disk.
constexpr size_t kMaxChunksPerBatch = 16;
constexpr size_t kBatchesInFlight = 2;

// A range of one unique file buffer. Replicated shards share a buffer, so one read fans out to every coord.
struct Chunk {
    uint64_t file_offset = 0;
    uint64_t buffer_offset = 0;
    uint64_t size = 0;
    const std::vector<MeshCoordinate>* coords = nullptr;
};

struct Batch {
    std::vector<Chunk> chunks;
    uint64_t staging_bytes = 0;
};

constexpr uint64_t align_down(uint64_t value, uint64_t alignment) { return value - (value % alignment); }

uint64_t read_window_bytes(const Chunk& chunk) {
    const uint64_t begin = align_down(chunk.file_offset, kDirectIoAlignment);
    return tt::align(chunk.file_offset + chunk.size, kDirectIoAlignment) - begin;
}

std::vector<std::byte> read_header(int fd, const std::string& file_name, uint64_t file_size) {
    uint64_t header_size = 0;
    TT_FATAL(file_size >= sizeof(header_size), "Tensor file \"{}\" is too small to be valid", file_name);
    TT_FATAL(
        pread(fd, &header_size, sizeof(header_size), 0) == static_cast<ssize_t>(sizeof(header_size)),
        "Failed to read header size from \"{}\"",
        file_name);
    TT_FATAL(
        sizeof(header_size) + header_size <= file_size,
        "Tensor file \"{}\" is truncated or corrupt (header_size={}, file_size={})",
        file_name,
        header_size,
        file_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header size is too large; this most likely indicates data corruption.");

    std::vector<std::byte> header(header_size);
    uint64_t done = 0;
    while (done < header_size) {
        const ssize_t n = pread(fd, header.data() + done, header_size - done, sizeof(header_size) + done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        TT_FATAL(n > 0, "Failed to read header from \"{}\": {}", file_name, n < 0 ? strerror(errno) : "end of file");
        done += n;
    }
    return header;
}

// Returns why the direct path cannot load this tensor, or nullopt if it can.
std::optional<std::string> unsupported_reason(
    const TensorFileDescription& description, const MeshDevice& device, uint64_t data_size) {
    if (description.mesh_shape != device.shape()) {
        return "file mesh shape differs from the device mesh";
    }
    std::unordered_set<MeshCoordinate> coords;
    for (const auto& shard : description.shards) {
        coords.insert(shard.coord);
    }
    for (const auto& coord : MeshCoordinateRange(device.shape())) {
        if (!coords.contains(coord)) {
            return "shards do not cover the whole mesh";
        }
    }
    if (description.spec.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return "memory config is not interleaved";
    }
    const uint64_t page_size = description.spec.compute_page_size_bytes();
    const uint64_t buffer_size = description.spec.compute_packed_buffer_size_bytes();
    if (page_size == 0 || buffer_size % page_size != 0) {
        return "buffer size is not a whole number of pages";
    }
    for (const auto& shard : description.shards) {
        if (shard.size != buffer_size) {
            return "a shard's size differs from the device buffer size";
        }
        if (shard.offset + shard.size > data_size) {
            return "a shard extends past the end of the file";
        }
    }
    return std::nullopt;
}

std::vector<Batch> plan_batches(
    const std::map<std::pair<uint64_t, uint64_t>, std::vector<MeshCoordinate>>& buffers,
    uint64_t data_offset,
    uint64_t page_size) {
    const uint64_t chunk_bytes = std::max(page_size, kTargetChunkBytes / page_size * page_size);

    std::vector<Chunk> chunks;
    for (uint64_t buffer_offset = 0;; buffer_offset += chunk_bytes) {
        bool any = false;
        for (const auto& [range, coords] : buffers) {
            const auto [offset, size] = range;
            if (buffer_offset >= size) {
                continue;
            }
            any = true;
            chunks.push_back(Chunk{
                .file_offset = data_offset + offset + buffer_offset,
                .buffer_offset = buffer_offset,
                .size = std::min(chunk_bytes, size - buffer_offset),
                .coords = &coords,
            });
        }
        if (!any) {
            break;
        }
    }

    std::vector<Batch> batches;
    for (size_t i = 0; i < chunks.size(); i += kMaxChunksPerBatch) {
        Batch batch;
        for (size_t j = i; j < std::min(chunks.size(), i + kMaxChunksPerBatch); ++j) {
            batch.staging_bytes += read_window_bytes(chunks[j]);
            batch.chunks.push_back(chunks[j]);
        }
        batches.push_back(std::move(batch));
    }
    return batches;
}

// Reads every chunk of `batch` into `staging` and returns a pointer to each chunk's first byte.
std::vector<std::byte*> read_batch(int fd, const std::string& file_name, const Batch& batch, std::byte* staging) {
    std::vector<std::byte*> data;
    data.reserve(batch.chunks.size());
    uint64_t position = 0;
    for (const auto& chunk : batch.chunks) {
        const uint64_t begin = align_down(chunk.file_offset, kDirectIoAlignment);
        const uint64_t window = read_window_bytes(chunk);
        const uint64_t needed = chunk.file_offset + chunk.size - begin;
        std::byte* dst = staging + position;
        // The window may run past the end of the file; the read then comes back short but still covers `needed`.
        uint64_t done = 0;
        while (done < needed) {
            const ssize_t n = pread(fd, dst + done, window - done, begin + done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            TT_FATAL(
                n > 0,
                "O_DIRECT read of \"{}\" at offset {} failed: {}",
                file_name,
                begin + done,
                n < 0 ? strerror(errno) : "end of file");
            done += n;
        }
        data.push_back(dst + (chunk.file_offset - begin));
        position += window;
    }
    return data;
}

struct AlignedFree {
    void operator()(std::byte* ptr) const { std::free(ptr); }
};

}  // namespace

Tensor load_tensor_flatbuffer_direct(const std::string& file_name, MeshDevice& device) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    auto close_fd = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    const uint64_t file_size = file_stat.st_size;

    const auto header = read_header(fd, file_name, file_size);
    const auto* header_bytes = reinterpret_cast<const uint8_t*>(header.data());
    flatbuffers::Verifier verifier(header_bytes, header.size());
    TT_FATAL(
        ttnn::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor data; this most likely indicates data corruption.");
    const auto description = describe_tensor_file(ttnn::flatbuffer::GetTensor(header_bytes));

    const uint64_t data_offset = sizeof(uint64_t) + header.size();
    if (auto reason = unsupported_reason(description, device, file_size - data_offset)) {
        log_debug(
            tt::LogAlways, "load_tensor_flatbuffer_direct: {} for \"{}\"; using the mmap path", *reason, file_name);
        return load_tensor_flatbuffer(file_name, &device);
    }

    int direct_fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct_fd == -1) {
        log_debug(
            tt::LogAlways,
            "load_tensor_flatbuffer_direct: O_DIRECT open of \"{}\" failed ({}); using the mmap path",
            file_name,
            strerror(errno));
        return load_tensor_flatbuffer(file_name, &device);
    }
    auto close_direct_fd = ttsl::make_cleanup([direct_fd]() { close(direct_fd); });

    // Unique file buffers, each with every coord it is written to.
    std::map<std::pair<uint64_t, uint64_t>, std::vector<MeshCoordinate>> buffers;
    for (const auto& shard : description.shards) {
        buffers[{shard.offset, shard.size}].push_back(shard.coord);
    }
    const auto batches = plan_batches(buffers, data_offset, description.spec.compute_page_size_bytes());

    uint64_t slot_bytes = kDirectIoAlignment;
    for (const auto& batch : batches) {
        slot_bytes = std::max(slot_bytes, batch.staging_bytes);
    }
    std::vector<std::unique_ptr<std::byte, AlignedFree>> slots;
    for (size_t i = 0; i < std::min(kBatchesInFlight, batches.size()); ++i) {
        auto* ptr = static_cast<std::byte*>(std::aligned_alloc(kDirectIoAlignment, slot_bytes));
        TT_FATAL(ptr != nullptr, "Failed to allocate {} bytes of staging memory", slot_bytes);
        slots.emplace_back(ptr);
    }

    auto mesh_tensor =
        tt::tt_metal::allocate_mesh_tensor_on_device_with_topology(device, description.spec, description.topology);
    // `enqueue_write_shards` takes a shared_ptr, so write through a non-owning view of the tensor's buffer.
    const auto& owner = mesh_tensor.mesh_buffer();
    auto target = MeshBuffer::create(owner.global_config(), owner.device_local_config(), &device, owner.address());

    struct ReadyBatch {
        size_t index = 0;
        size_t slot = 0;
        std::vector<std::byte*> data;
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<size_t> free_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        free_slots.push_back(i);
    }
    std::deque<ReadyBatch> ready;
    std::exception_ptr reader_error;
    bool stop = false;

    std::thread reader([&]() {
        try {
            for (size_t index = 0; index < batches.size(); ++index) {
                size_t slot = 0;
                {
                    std::unique_lock lock(mutex);
                    cv.wait(lock, [&]() { return stop || !free_slots.empty(); });
                    if (stop) {
                        return;
                    }
                    slot = free_slots.front();
                    free_slots.pop_front();
                }
                auto data = read_batch(direct_fd, file_name, batches[index], slots[slot].get());
                {
                    std::lock_guard lock(mutex);
                    ready.push_back(ReadyBatch{.index = index, .slot = slot, .data = std::move(data)});
                }
                cv.notify_all();
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            reader_error = std::current_exception();
            cv.notify_all();
        }
    });
    auto join_reader = ttsl::make_cleanup([&]() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        cv.notify_all();
        reader.join();
    });

    auto& cq = device.mesh_command_queue();
    for (size_t index = 0; index < batches.size(); ++index) {
        ReadyBatch batch;
        {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&]() { return reader_error != nullptr || !ready.empty(); });
            if (reader_error != nullptr) {
                std::rethrow_exception(reader_error);
            }
            batch = std::move(ready.front());
            ready.pop_front();
        }

        std::vector<ShardDataTransfer> transfers;
        const auto& chunks = batches[batch.index].chunks;
        for (size_t i = 0; i < chunks.size(); ++i) {
            for (const auto& coord : *chunks[i].coords) {
                transfers.push_back(ShardDataTransfer{coord}
                                        .host_data(batch.data[i])
                                        .region(tt::tt_metal::BufferRegion(chunks[i].buffer_offset, chunks[i].size)));
            }
        }
        // Without pinned memory this copies every transfer into the command queue before returning, so the staging
        // slot can be refilled immediately; later work on this queue is ordered after the writes.
        cq.enqueue_write_shards(target, transfers, /*blocking=*/false);

        {
            std::lock_guard lock(mutex);
            free_slots.push_back(batch.slot);
        }
        cv.notify_all();
    }

    return Tensor(std::move(mesh_tensor));
}

}  // namespace ttnn
