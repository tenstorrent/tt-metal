// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "ttnn/types.hpp"

namespace ttnn::global_semaphore {

// Single Device Creation API
GlobalSemaphore create_global_semaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

// MeshDevice Creation API
GlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1);

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore);

void reset_global_semaphore_value(const GlobalSemaphore& global_semaphore, uint32_t reset_value);

}  // namespace ttnn::global_semaphore
