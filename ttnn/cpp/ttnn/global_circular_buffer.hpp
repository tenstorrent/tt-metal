// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

struct MultiDeviceGlobalCircularBuffer {
    MultiDeviceGlobalCircularBuffer(MeshDevice* mesh_device);
    std::vector<std::shared_ptr<GlobalCircularBuffer>> global_circular_buffers;
};

// Single Device APIs
std::shared_ptr<GlobalCircularBuffer> create_global_circular_buffer(
    Device* device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Multi Device APIs
MultiDeviceGlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::global_circular_buffer
