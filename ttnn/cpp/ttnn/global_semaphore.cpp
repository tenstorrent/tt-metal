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

tt::tt_metal::DeviceAddr get_global_semaphore_address(const std::shared_ptr<GlobalSemaphore>& global_semaphore) {
    auto* device = global_semaphore->device();
    tt::tt_metal::DeviceAddr address = 0;
    device->push_work([&global_semaphore, &address] { address = global_semaphore->address(); }, /*blocking=*/true);
    return address;
}

void reset_global_semaphore_value(
    const std::shared_ptr<GlobalSemaphore>& global_semaphore,
    uint32_t reset_value,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto* device = global_semaphore->device();
    device->push_work([global_semaphore, reset_value, sub_device_ids] {
        global_semaphore->reset_semaphore_value(reset_value, sub_device_ids);
    });
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
MultiDeviceGlobalSemaphore create_global_semaphore_with_same_address(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    uint32_t attempts,
    bool search_max) {
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
    auto global_semaphores = multi_device_global_semaphore.global_semaphores;
    auto first_addr = global_semaphores.front()->address();
    bool all_same = std::all_of(global_semaphores.begin(), global_semaphores.end(), [first_addr](const auto& sem) {
        return sem->address() == first_addr;
    });

    if (!all_same) {
        tt::log_debug("chkpt 1, attempts: {}", attempts);
        DeviceAddr target_addr = global_semaphores.front()->address();
        for (auto i = 1; i < global_semaphores.size(); i++) {
            tt::log_debug("chkpt 1.1, i: {}, global_semaphores[i]->address(): {}", i, global_semaphores[i]->address());
            if (search_max) {
                if (global_semaphores[i]->address() > target_addr) {
                    target_addr = global_semaphores[i]->address();
                }
            } else {
                if (global_semaphores[i]->address() < target_addr) {
                    target_addr = global_semaphores[i]->address();
                }
            }
        };
        tt::log_debug("chkpt 2, target_addr: {}", target_addr);
        for (auto i = 0; i < global_semaphores.size(); i++) {
            auto* device = devices[i];
            auto& global_semaphore = multi_device_global_semaphore.global_semaphores[i];
            tt::log_debug("pushed, i: {}", i);
            device->push_work([i,
                               device,
                               attempts,
                               target_addr,
                               &cores,
                               initial_value,
                               buffer_type,
                               sub_device_ids,
                               &global_semaphore] {
                size_t attempt = 0;
                std::vector<std::shared_ptr<tt::tt_metal::GlobalSemaphore>> garbage;
                tt::log_debug("global_semaphore->address(): {}", global_semaphore->address());
                while (global_semaphore->address() != target_addr) {
                    auto sem = GlobalSemaphore::create(device, cores, initial_value, buffer_type, sub_device_ids);

                    if (i == 0) {
                        tt::log_debug("chkpt 3, sem->address(): {}", sem->address());
                    }

                    if (sem->address() == target_addr) {
                        global_semaphore = sem;
                    } else {
                        garbage.push_back(std::move(sem));
                        attempt++;
                    }

                    if (attempt > attempts) {
                        TT_THROW("Failed to create global semaphores with the same address");
                    }
                }
            });
        }
        for (auto device : devices) {
            device->synchronize();
        }
    }

    return multi_device_global_semaphore;
}
std::vector<tt::tt_metal::DeviceAddr> get_global_semaphore_address(const MultiDeviceGlobalSemaphore& global_semaphore) {
    std::vector<tt::tt_metal::DeviceAddr> addresses(global_semaphore.global_semaphores.size());
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
    const MultiDeviceGlobalSemaphore& global_semaphore,
    uint32_t reset_value,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    for (const auto& global_semaphore : global_semaphore.global_semaphores) {
        reset_global_semaphore_value(global_semaphore, reset_value, sub_device_ids);
    }
}

}  // namespace ttnn::global_semaphore
