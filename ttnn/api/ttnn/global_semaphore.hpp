// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "ttnn/types.hpp"

namespace ttnn::global_semaphore {

struct MultiDeviceGlobalSemaphore {
    MultiDeviceGlobalSemaphore(size_t num_devices);
    std::vector<GlobalSemaphore> global_semaphores;

    static constexpr auto attribute_names = std::forward_as_tuple("global_semaphores");
    auto attribute_values() const { return std::forward_as_tuple(this->global_semaphores); }
};

// Single Device Creation API
GlobalSemaphore create_global_semaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

// MeshDevice Creation API
GlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1);

// Multi-device Creation API
MultiDeviceGlobalSemaphore create_global_semaphore(
    const std::vector<IDevice*>& devices,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1);
MultiDeviceGlobalSemaphore create_global_semaphore_with_same_address(
    const std::vector<IDevice*>& devices,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type,
    uint32_t attempts,
    bool search_max = false);

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore);
std::vector<tt::tt_metal::DeviceAddr> get_global_semaphore_address(const MultiDeviceGlobalSemaphore& global_semaphore);

void reset_global_semaphore_value(const GlobalSemaphore& global_semaphore, uint32_t reset_value);
void reset_global_semaphore_value(const MultiDeviceGlobalSemaphore& global_semaphore, uint32_t reset_value);

}  // namespace ttnn::global_semaphore
