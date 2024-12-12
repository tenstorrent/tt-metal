// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_circular_buffer.hpp"

#include <memory>
#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::global_circular_buffer {

MultiDeviceGlobalCircularBuffer::MultiDeviceGlobalCircularBuffer(MeshDevice* mesh_device) {
    TT_ASSERT(
        mesh_device != nullptr,
        "Must provide a valid mesh_device when initializing a global circular buffer on multiple devices.");
    this->global_circular_buffers = std::vector<std::shared_ptr<GlobalCircularBuffer>>(mesh_device->num_devices());
}

std::shared_ptr<GlobalCircularBuffer> create_global_circular_buffer(
    Device* device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::shared_ptr<GlobalCircularBuffer> global_cb;
    device->push_work(
        [device, &sender_receiver_core_mapping, size, buffer_type, sub_device_ids, &global_cb]() {
            global_cb = tt::tt_metal::v1::experimental::CreateGlobalCircularBuffer(
                device, sender_receiver_core_mapping, size, buffer_type, sub_device_ids);
        },
        /*blocking=*/true);
    return global_cb;
}

MultiDeviceGlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    MultiDeviceGlobalCircularBuffer multi_device_global_cb(mesh_device);
    const auto& devices = mesh_device->get_devices();
    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto* device = devices[i];
        auto& global_cb = multi_device_global_cb.global_circular_buffers[i];
        device->push_work([device, &sender_receiver_core_mapping, size, buffer_type, sub_device_ids, &global_cb]() {
            global_cb = tt::tt_metal::v1::experimental::CreateGlobalCircularBuffer(
                device, sender_receiver_core_mapping, size, buffer_type, sub_device_ids);
        });
    }
    for (auto* device : devices) {
        device->synchronize();
    }
    return multi_device_global_cb;
}

}  // namespace ttnn::global_circular_buffer
