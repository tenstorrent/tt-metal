// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include <memory>
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace ttnn::global_semaphore {

MultiDeviceGlobalSemaphore::MultiDeviceGlobalSemaphore(MeshDevice* mesh_device) {
    TT_ASSERT(
        mesh_device != nullptr,
        "Must provide a valid mesh_device when initializing a global semaphore on multiple devices.");
    this->global_semaphores.reserve(mesh_device->num_devices());
}

GlobalSemaphore create_global_semaphore(
    Device* device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    return CreateGlobalSemaphore(device, cores, initial_value, buffer_type, sub_device_ids);
}

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore) {
    return global_semaphore.address();
}

void reset_global_semaphore_value(
    const GlobalSemaphore& global_semaphore, uint32_t reset_value, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    global_semaphore.reset_semaphore_value(reset_value, sub_device_ids);
}

MultiDeviceGlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    MultiDeviceGlobalSemaphore multi_device_global_semaphore(mesh_device);
    auto& global_semaphores = multi_device_global_semaphore.global_semaphores;
    const auto& devices = mesh_device->get_devices();
    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto* device = devices[i];
        global_semaphores.push_back(create_global_semaphore(device, cores, initial_value, buffer_type, sub_device_ids));
    }
    return multi_device_global_semaphore;
}
std::vector<tt::tt_metal::DeviceAddr> get_global_semaphore_address(const MultiDeviceGlobalSemaphore& global_semaphore) {
    std::vector<tt::tt_metal::DeviceAddr> addresses(global_semaphore.global_semaphores.size());
    const auto& global_semaphores = global_semaphore.global_semaphores;
    for (uint32_t i = 0; i < global_semaphores.size(); ++i) {
        addresses[i] = get_global_semaphore_address(global_semaphores[i]);
    }
    return addresses;
}

void reset_global_semaphore_value(
    const MultiDeviceGlobalSemaphore& global_semaphore,
    uint32_t reset_value,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    for (const auto& global_semaphore : global_semaphore.global_semaphores) {
        reset_global_semaphore_value(global_semaphore, reset_value, sub_device_ids);
    }
}

}  // namespace ttnn::global_semaphore
