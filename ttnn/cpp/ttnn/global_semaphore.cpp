// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::global_semaphore {

GlobalSemaphore create_global_semaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    return CreateGlobalSemaphore(device, cores, initial_value, buffer_type);
}

GlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    return CreateGlobalSemaphore(mesh_device, cores, initial_value, buffer_type);
}

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore) {
    return global_semaphore.address();
}

void reset_global_semaphore_value(const GlobalSemaphore& global_semaphore, uint32_t reset_value) {
    global_semaphore.reset_semaphore_value(reset_value);
}

}  // namespace ttnn::global_semaphore
