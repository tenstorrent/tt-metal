// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_circular_buffer.hpp"

#include <memory>
#include <tt-metalium/global_circular_buffer_impl.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::global_circular_buffer {

MultiDeviceGlobalCircularBuffer::MultiDeviceGlobalCircularBuffer(MeshDevice* mesh_device) {
    TT_ASSERT(
        mesh_device != nullptr,
        "Must provide a valid mesh_device when initializing a global circular buffer on multiple devices.");
    this->global_circular_buffers.reserve(mesh_device->num_devices());
}

GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

MultiDeviceGlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    MultiDeviceGlobalCircularBuffer multi_device_global_cb(mesh_device);
    auto& global_circular_buffers = multi_device_global_cb.global_circular_buffers;
    const auto& devices = mesh_device->get_devices();
    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto* device = devices[i];
        global_circular_buffers.push_back(
            create_global_circular_buffer(device, sender_receiver_core_mapping, size, buffer_type));
    }
    return multi_device_global_cb;
}

}  // namespace ttnn::global_circular_buffer
