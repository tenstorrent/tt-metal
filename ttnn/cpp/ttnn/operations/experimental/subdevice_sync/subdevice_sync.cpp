// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/subdevice_sync/subdevice_sync.hpp"
#include "device/subdevice_sync_device_operation.hpp"

#include <tuple>

#include <tt_stl/assert.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::operations::experimental {

using subdevice_sync::SubdeviceSyncMode;

namespace {
ttnn::MeshDevice& semaphore_mesh_device(const tt::tt_metal::GlobalSemaphore& semaphore) {
    auto* mesh_device = dynamic_cast<ttnn::MeshDevice*>(semaphore.device());
    TT_FATAL(mesh_device != nullptr, "subdevice_sync: the semaphore must be created on a MeshDevice");
    return *mesh_device;
}

void check_on_semaphore(const tt::tt_metal::GlobalSemaphore& semaphore, const CoreCoord& core, const char* what) {
    const CoreRangeSet sem_cores = std::get<0>(semaphore.attribute_values());
    TT_FATAL(sem_cores.contains(core), "{}: core {} is not one of the semaphore's cores {}", what, core, sem_cores);
}
}  // namespace

void subdevice_signal(
    const tt::tt_metal::GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    const CoreCoord& core,
    const CoreCoord& consumer_core,
    uint32_t increment) {
    check_on_semaphore(semaphore, consumer_core, "subdevice_signal(consumer_core)");
    ttnn::prim::subdevice_sync(
        semaphore_mesh_device(semaphore),
        SubdeviceSyncMode::Signal,
        static_cast<uint32_t>(semaphore.address()),
        core,
        consumer_core,
        increment,
        0,
        subdevice_id);
}

void subdevice_wait(
    const tt::tt_metal::GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    const CoreCoord& core,
    uint32_t value,
    uint32_t timeout_cycles) {
    check_on_semaphore(semaphore, core, "subdevice_wait(core)");
    ttnn::prim::subdevice_sync(
        semaphore_mesh_device(semaphore),
        SubdeviceSyncMode::Wait,
        static_cast<uint32_t>(semaphore.address()),
        core,
        core,
        value,
        timeout_cycles,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental
