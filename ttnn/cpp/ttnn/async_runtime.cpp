// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "async_runtime.hpp"

#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"

namespace ttnn {
using DeviceBuffer = std::shared_ptr<Buffer>;
using queue_id = uint8_t;

DeviceBuffer allocate_interleaved_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    uint32_t page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, buffer_size_bytes, shape, tile);
    return Buffer::create(device, buffer_size_bytes, page_size, memory_config.buffer_type);
}

DeviceBuffer allocate_contiguous_buffer_on_device(
    size_t buffer_size_bytes, Device* device, const MemoryConfig& memory_config) {
    return Buffer::create(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_type);
}

DeviceBuffer allocate_sharded_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const ShardSpecBuffer& shard_params,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    tt::tt_metal::tensor_impl::validate_sharded_buffer_allocation(
        shape, layout, data_type, shard_params, memory_config, tile);
    const auto& page_shape = shard_params.page_shape;
    uint32_t size_of_element = tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    uint32_t page_size = page_shape[0] * page_shape[1] * size_of_element;
    if (layout == Layout::TILE) {
        page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, buffer_size_bytes, shape, tile);
    }

    return Buffer::create(
        device, buffer_size_bytes, page_size, memory_config.buffer_type, memory_config.memory_layout, shard_params);
}

DeviceBuffer allocate_buffer_on_device(
    size_t buffer_size_bytes,
    types::Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<ShardSpecBuffer>& shard_spec,
    const std::optional<Tile>& tile) {
    if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return allocate_interleaved_buffer_on_device(
            buffer_size_bytes, device, shape, data_type, layout, memory_config, tile);
    } else if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::SINGLE_BANK) {
        return allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    } else {
        return allocate_sharded_buffer_on_device(
            buffer_size_bytes, device, shape, data_type, layout, shard_spec.value(), memory_config, tile);
    }
}

void write_buffer(
    queue_id cq_id,
    Tensor& dst,
    std::vector<std::shared_ptr<void>> src,
    const std::optional<std::size_t> transfer_size) {
    uint32_t dst_ref_count = dst.tensor_attributes->record_main_thread_ref_count();
    for (const auto worker : dst.get_workers()) {
        auto src_for_device = (src.size() == 1) ? src.at(0) : src.at(worker->id());
        worker->push_work([worker, src_for_device, dst, cq_id, transfer_size]() {
            auto shard = tt::tt_metal::get_shard_for_device(dst, worker);
            tt::tt_metal::memcpy(worker->command_queue(cq_id), shard, src_for_device.get(), transfer_size);
        });
    }
    dst.tensor_attributes->update_main_thread_ref_count(dst.workers.at(0), dst_ref_count);
}

void read_buffer(
    queue_id cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<std::size_t> transfer_size,
    size_t src_offset,
    bool blocking) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    uint32_t src_ref_count = src.tensor_attributes->record_main_thread_ref_count();
    for (const auto worker : src.get_workers()) {
        auto dst_for_device = (dst.size() == 1) ? dst.at(0) : dst.at(worker->id());
        worker->push_work([worker, dst_for_device, src, cq_id, transfer_size, src_offset, blocking]() {
            const auto& shard = tt::tt_metal::get_shard_for_device(src, worker);
            tt::tt_metal::memcpy(worker->command_queue(cq_id), dst_for_device.get(), shard, transfer_size, blocking);
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

void event_synchronize(std::shared_ptr<Event> event) { EventSynchronize(event); }

bool event_query(std::shared_ptr<Event> event) { return EventQuery(event); }

void wait_for_event(CommandQueue& cq, std::shared_ptr<Event> event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    cq_worker->push_work([cq_worker, cq_id, event]() { EnqueueWaitForEvent(cq_worker->command_queue(cq_id), event); });
}

void record_event(CommandQueue& cq, std::shared_ptr<Event> event) {
    auto cq_id = cq.id();
    auto cq_worker = cq.device();
    cq_worker->push_work([cq_worker, cq_id, event]() { EnqueueRecordEvent(cq_worker->command_queue(cq_id), event); });
}

}  // namespace ttnn
