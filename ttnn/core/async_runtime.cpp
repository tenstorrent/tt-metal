// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/async_runtime.hpp"

#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/distributed/api.hpp"

using namespace tt::tt_metal;

namespace ttnn {

void write_buffer(
    QueueId cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<BufferRegion>& region) {
    if (auto mesh_device = dst.mesh_device()) {
        auto& cq = mesh_device->mesh_command_queue(*cq_id);
        auto device_tensors = ttnn::distributed::get_device_tensors(dst);
        for (size_t i = 0; i < device_tensors.size(); i++) {
            tt::tt_metal::memcpy(cq, device_tensors[i], src.at(i).get(), region);
        }
    } else {
        auto* dst_device = dst.device();
        auto src_for_device = (src.size() == 1) ? src.at(0) : src.at(dst_device->id());
        tt::tt_metal::memcpy(dst_device->command_queue(*cq_id), dst, src_for_device.get(), region);
    }
}

void read_buffer(
    QueueId cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<BufferRegion>& region,
    size_t src_offset,
    bool blocking) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    if (auto mesh_device = src.mesh_device()) {
        auto& cq = mesh_device->mesh_command_queue(*cq_id);
        auto device_tensors = ttnn::distributed::get_device_tensors(src);
        for (size_t i = 0; i < device_tensors.size(); i++) {
            tt::tt_metal::memcpy(cq, dst.at(i).get(), device_tensors[i], region);
        }
    } else {
        auto* src_device = src.device();
        auto dst_for_device = (dst.size() == 1) ? dst.at(0) : dst.at(src_device->id());
        tt::tt_metal::memcpy(src_device->command_queue(*cq_id), dst_for_device.get(), src, region, blocking);
    }
}

void queue_synchronize(CommandQueue& cq) {
    // Wait for device CQ to finish
    Finish(cq);
}
void queue_synchronize(tt::tt_metal::distributed::MeshCommandQueue& cq) { cq.finish(); }

void event_synchronize(const std::shared_ptr<Event>& event) { EventSynchronize(event); }
void event_synchronize(const tt::tt_metal::distributed::MeshEvent& event) {
    tt::tt_metal::distributed::EventSynchronize(event);
}

void wait_for_event(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    EnqueueWaitForEvent(cq_worker->command_queue(cq_id), event);
}
void wait_for_event(
    tt::tt_metal::distributed::MeshCommandQueue& cq, const tt::tt_metal::distributed::MeshEvent& event) {
    tt::tt_metal::distributed::EnqueueWaitForEvent(cq, event);
}

void record_event(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    EnqueueRecordEvent(cq_worker->command_queue(cq_id), event);
}
tt::tt_metal::distributed::MeshEvent record_event(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return tt::tt_metal::distributed::EnqueueRecordEvent(cq);
}
tt::tt_metal::distributed::MeshEvent record_event_to_host(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return tt::tt_metal::distributed::EnqueueRecordEventToHost(cq);
}

}  // namespace ttnn
