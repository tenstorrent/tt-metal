// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/async_runtime.hpp"

#include "ttnn/distributed/api.hpp"

using namespace tt::tt_metal;

namespace ttnn {

void write_buffer(
    QueueId cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<BufferRegion>& region) {
    auto* mesh_device = dst.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    auto& cq = mesh_device->mesh_command_queue(*cq_id);
    auto device_tensors = ttnn::distributed::get_device_tensors(dst);
    for (size_t i = 0; i < device_tensors.size(); i++) {
        tt::tt_metal::memcpy(cq, device_tensors[i], src.at(i).get(), region);
    }
}

void read_buffer(
    QueueId cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<BufferRegion>& region,
    size_t src_offset,
    bool /*blocking*/) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    auto* mesh_device = src.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    auto& cq = mesh_device->mesh_command_queue(*cq_id);
    auto device_tensors = ttnn::distributed::get_device_tensors(src);
    for (size_t i = 0; i < device_tensors.size(); i++) {
        tt::tt_metal::memcpy(cq, dst.at(i).get(), device_tensors[i], region);
    }
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
