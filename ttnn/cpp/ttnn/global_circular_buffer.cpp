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
    this->global_circular_buffers.reserve(mesh_device->num_devices());
}

GlobalCircularBuffer create_global_circular_buffer(
    Device* device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    return tt::tt_metal::v1::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type, sub_device_ids);
}

MultiDeviceGlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    MultiDeviceGlobalCircularBuffer multi_device_global_cb(mesh_device);
    auto& global_circular_buffers = multi_device_global_cb.global_circular_buffers;
    const auto& devices = mesh_device->get_devices();
    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto* device = devices[i];
        global_circular_buffers.push_back(
            create_global_circular_buffer(device, sender_receiver_core_mapping, size, buffer_type, sub_device_ids));
    }
    return multi_device_global_cb;
}

}  // namespace ttnn::global_circular_buffer
