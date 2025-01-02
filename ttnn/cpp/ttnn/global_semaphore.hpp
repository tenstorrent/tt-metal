// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/host_api.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_semaphore {

struct MultiDeviceGlobalSemaphore {
    MultiDeviceGlobalSemaphore(MeshDevice* mesh_device);
    std::vector<GlobalSemaphore> global_semaphores;
};

// Single Device APIs
GlobalSemaphore create_global_semaphore(
    Device* device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

tt::tt_metal::DeviceAddr get_global_semaphore_address(const GlobalSemaphore& global_semaphore);

void reset_global_semaphore_value(
    const GlobalSemaphore& global_semaphore,
    uint32_t reset_value,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

// Multi Device APIs
MultiDeviceGlobalSemaphore create_global_semaphore(
    MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

std::vector<tt::tt_metal::DeviceAddr> get_global_semaphore_address(const MultiDeviceGlobalSemaphore& global_semaphore);

void reset_global_semaphore_value(
    const MultiDeviceGlobalSemaphore& global_semaphore,
    uint32_t reset_value,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

}  // namespace ttnn::global_semaphore
