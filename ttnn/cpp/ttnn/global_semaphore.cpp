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
    this->global_semaphores = std::vector<std::shared_ptr<GlobalSemaphore>>(mesh_device->num_devices());
}

std::shared_ptr<GlobalSemaphore> create_global_semaphore(
    Device* device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::shared_ptr<GlobalSemaphore> global_semaphore = nullptr;
    device->push_work(
        [device, &cores, initial_value, buffer_type, sub_device_ids, &global_semaphore] {
            global_semaphore = GlobalSemaphore::create(device, cores, initial_value, buffer_type, sub_device_ids);
        },
        /*blocking=*/true);
    return global_semaphore;
}

DeviceAddr get_global_semaphore_address(const std::shared_ptr<GlobalSemaphore>& global_semaphore) {
    auto* device = global_semaphore->device();
    DeviceAddr address = 0;
    device->push_work([&global_semaphore, &address] { address = global_semaphore->address(); }, /*blocking=*/true);
    return address;
}

void reset_global_semaphore_value(
    const std::shared_ptr<GlobalSemaphore>& global_semaphore, const std::vector<SubDeviceId>& sub_device_ids) {
    auto* device = global_semaphore->device();
    device->push_work([global_semaphore, sub_device_ids] { global_semaphore->reset_semaphore_value(sub_device_ids); });
}

MultiDeviceGlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    MultiDeviceGlobalSemaphore multi_device_global_semaphore(mesh_device);
    const auto& devices = mesh_device->get_devices();
    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto* device = devices[i];
        auto& global_semaphore = multi_device_global_semaphore.global_semaphores[i];
        device->push_work([device, &cores, initial_value, buffer_type, sub_device_ids, &global_semaphore] {
            global_semaphore = GlobalSemaphore::create(device, cores, initial_value, buffer_type, sub_device_ids);
        });
    }
    for (auto device : devices) {
        device->synchronize();
    }
    return multi_device_global_semaphore;
}
std::vector<DeviceAddr> get_global_semaphore_address(const MultiDeviceGlobalSemaphore& global_semaphore) {
    std::vector<DeviceAddr> addresses(global_semaphore.global_semaphores.size());
    const auto& global_semaphores = global_semaphore.global_semaphores;
    for (uint32_t i = 0; i < global_semaphores.size(); ++i) {
        const auto& global_semaphore = global_semaphores[i];
        auto& address = addresses[i];
        auto* device = global_semaphore->device();
        device->push_work([&global_semaphore, &address] { address = global_semaphore->address(); });
    }
    for (const auto& global_semaphore : global_semaphores) {
        global_semaphore->device()->synchronize();
    }
    return addresses;
}

void reset_global_semaphore_value(
    const MultiDeviceGlobalSemaphore& global_semaphore, const std::vector<SubDeviceId>& sub_device_ids) {
    for (const auto& global_semaphore : global_semaphore.global_semaphores) {
        reset_global_semaphore_value(global_semaphore, sub_device_ids);
    }
}

}  // namespace ttnn::global_semaphore
