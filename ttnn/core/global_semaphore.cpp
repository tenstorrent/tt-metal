// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::global_semaphore {

GlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    auto guard = tt::tt_metal::make_allocation_context_guard("ttnn.create_global_semaphore");
    return GlobalSemaphore(*mesh_device, cores, initial_value, buffer_type);
}

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore) {
    return global_semaphore.address();
}

void reset_global_semaphore_value(const GlobalSemaphore& global_semaphore, uint32_t reset_value) {
    global_semaphore.reset_semaphore_value(reset_value);
}

}  // namespace ttnn::global_semaphore
