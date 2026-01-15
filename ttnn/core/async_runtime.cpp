// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/async_runtime.hpp"

#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/api.hpp"
using namespace tt::tt_metal;

namespace ttnn {

void write_buffer(
    QueueId cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<BufferRegion>& region) {
    auto* mesh_device = dst.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    auto& cq = mesh_device->mesh_command_queue(*cq_id);

    // Build shard data transfers for all coordinates in the mesh (following WriteShard pattern)
    std::vector<tt::tt_metal::distributed::ShardDataTransfer> shard_data_transfers;
    shard_data_transfers.reserve(mesh_device->shape().mesh_size());

    size_t i = 0;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        auto transfer = tt::tt_metal::distributed::ShardDataTransfer{coord}.host_data(src.at(i).get());
        if (region.has_value()) {
            transfer.region(region);
        }
        shard_data_transfers.push_back(std::move(transfer));
        ++i;
    }
    cq.enqueue_write_shards(dst.mesh_buffer(), shard_data_transfers, false);
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
    auto& cq = mesh_device->mesh_command_queue(*cq_id);

    // Build shard data transfers for all coordinates in the mesh (following ReadShard pattern)
    std::vector<tt::tt_metal::distributed::ShardDataTransfer> shard_data_transfers;
    shard_data_transfers.reserve(mesh_device->shape().mesh_size());

    size_t i = 0;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        auto transfer = tt::tt_metal::distributed::ShardDataTransfer{coord}.host_data(dst.at(i).get());
        if (region.has_value()) {
            transfer.region(region);
        }
        shard_data_transfers.push_back(std::move(transfer));
        ++i;
    }
    cq.enqueue_read_shards(shard_data_transfers, src.mesh_buffer(), blocking);
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
