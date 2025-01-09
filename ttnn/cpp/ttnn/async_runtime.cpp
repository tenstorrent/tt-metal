// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "async_runtime.hpp"

#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"

using namespace tt::tt_metal;

namespace ttnn {

void write_buffer(
    queue_id cq_id,
    Tensor& dst,
    std::vector<std::shared_ptr<void>> src,
    const std::optional<DeviceBufferRegion>& region) {
    uint32_t dst_ref_count = dst.tensor_attributes->record_main_thread_ref_count();
    for (const auto worker : dst.get_workers()) {
        auto src_for_device = (src.size() == 1) ? src.at(0) : src.at(worker->id());
        worker->push_work([worker, src_for_device, dst, cq_id, region]() {
            auto shard = tt::tt_metal::get_shard_for_device(dst, worker);
            tt::tt_metal::memcpy(worker->command_queue(cq_id), shard, src_for_device.get(), region);
        });
    }
    dst.tensor_attributes->update_main_thread_ref_count(dst.workers.at(0), dst_ref_count);
}

void read_buffer(
    queue_id cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<DeviceBufferRegion>& region,
    size_t src_offset,
    bool blocking) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    uint32_t src_ref_count = src.tensor_attributes->record_main_thread_ref_count();
    for (const auto worker : src.get_workers()) {
        auto dst_for_device = (dst.size() == 1) ? dst.at(0) : dst.at(worker->id());
        worker->push_work([worker, dst_for_device, src, cq_id, region, src_offset, blocking]() {
            const auto& shard = tt::tt_metal::get_shard_for_device(src, worker);
            tt::tt_metal::memcpy(worker->command_queue(cq_id), dst_for_device.get(), shard, region, blocking);
        });
    }
    if (blocking) {
        for (auto worker : src.get_workers()) {
            worker->synchronize();
        }
    }
    src.tensor_attributes->update_main_thread_ref_count(src.workers.at(0), src_ref_count);
}

void queue_synchronize(CommandQueue& cq) {
    // Ensure that all work pushed to async engine has been passed
    // off to device CQ
    cq.device()->synchronize();
    // Wait for device CQ to finish
    Finish(cq);
}

void event_synchronize(const std::shared_ptr<Event>& event) { EventSynchronize(event); }

bool event_query(const std::shared_ptr<Event>& event) { return EventQuery(event); }

void wait_for_event(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    cq_worker->push_work([cq_worker, cq_id, event]() { EnqueueWaitForEvent(cq_worker->command_queue(cq_id), event); });
}

void record_event(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    cq_worker->push_work([cq_worker, cq_id, event]() { EnqueueRecordEvent(cq_worker->command_queue(cq_id), event); });
}

}  // namespace ttnn
