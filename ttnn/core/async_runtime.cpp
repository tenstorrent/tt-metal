// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/async_runtime.hpp"

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>

#include "ttnn/distributed/api.hpp"

using namespace tt::tt_metal;

namespace ttnn {

void write_buffer(
    QueueId cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<BufferRegion>& region) {
    auto* mesh_device = dst.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    TT_FATAL(dst.storage_type() == StorageType::DEVICE, "Destination tensor must be on device");

    const auto& device_storage = dst.device_storage();
    const auto& coords = device_storage.coords;
    TT_FATAL(
        src.size() == coords.size(),
        "Number of source buffers ({}) must match number of device shards ({})",
        src.size(),
        coords.size());

    auto mesh_buffer = device_storage.mesh_buffer;
    TT_FATAL(mesh_buffer != nullptr, "Destination tensor must have allocated buffer");

    // Determine the size of each shard
    const size_t shard_size_bytes =
        region.has_value() ? region->size : dst.tensor_spec().compute_packed_buffer_size_bytes();

    // Create a DistributedHostBuffer with HostBuffers borrowing from the user's memory
    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device->shape());
    for (size_t i = 0; i < coords.size(); i++) {
        const auto& coord = coords[i];
        const auto& user_buffer = src[i];

        distributed_host_buffer.emplace_shard(coord, [&]() {
            // Create a MemoryPin from the shared_ptr to keep the user's data alive
            MemoryPin pin(user_buffer);
            // Create a Span pointing to the user's data
            tt::stl::Span<std::byte> span(static_cast<std::byte*>(user_buffer.get()), shard_size_bytes);
            // Create a HostBuffer that borrows the user's memory
            return HostBuffer(span, std::move(pin));
        });
    }

    auto& cq = mesh_device->mesh_command_queue(*cq_id);
    cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
}

void read_buffer(
    QueueId cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<BufferRegion>& region,
    size_t src_offset,
    bool blocking) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    auto* mesh_device = src.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    TT_FATAL(src.storage_type() == StorageType::DEVICE, "Source tensor must be on device");

    const auto& device_storage = src.device_storage();
    const auto& coords = device_storage.coords;
    TT_FATAL(
        dst.size() == coords.size(),
        "Number of destination buffers ({}) must match number of device shards ({})",
        dst.size(),
        coords.size());

    auto mesh_buffer = device_storage.mesh_buffer;
    TT_FATAL(mesh_buffer != nullptr, "Source tensor must have allocated buffer");

    // Determine the size of each shard
    const size_t shard_size_bytes =
        region.has_value() ? region->size : src.tensor_spec().compute_packed_buffer_size_bytes();

    // Create a DistributedHostBuffer with HostBuffers borrowing from the user's memory
    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device->shape());
    for (size_t i = 0; i < coords.size(); i++) {
        const auto& coord = coords[i];
        const auto& user_buffer = dst[i];

        distributed_host_buffer.emplace_shard(coord, [&]() {
            // Create a MemoryPin from the shared_ptr to keep the user's data alive
            MemoryPin pin(user_buffer);
            // Create a Span pointing to the user's data
            tt::stl::Span<std::byte> span(static_cast<std::byte*>(user_buffer.get()), shard_size_bytes);
            // Create a HostBuffer that borrows the user's memory
            return HostBuffer(span, std::move(pin));
        });
    }

    auto& cq = mesh_device->mesh_command_queue(*cq_id);
    // Convert coords to unordered_set for the API
    std::unordered_set<distributed::MeshCoordinate> shard_coords(coords.begin(), coords.end());
    cq.enqueue_read(mesh_buffer, distributed_host_buffer, shard_coords, blocking);
}

void queue_synchronize(tt::tt_metal::distributed::MeshCommandQueue& cq) { cq.finish(); }

void event_synchronize(const tt::tt_metal::distributed::MeshEvent& event) {
    tt::tt_metal::distributed::EventSynchronize(event);
}

void wait_for_event(
    tt::tt_metal::distributed::MeshCommandQueue& cq, const tt::tt_metal::distributed::MeshEvent& event) {
    cq.enqueue_wait_for_event(event);
}

tt::tt_metal::distributed::MeshEvent record_event(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return cq.enqueue_record_event();
}
tt::tt_metal::distributed::MeshEvent record_event_to_host(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return cq.enqueue_record_event_to_host();
}

}  // namespace ttnn
