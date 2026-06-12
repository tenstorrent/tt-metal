// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <algorithm>
#include <memory>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::global_circular_buffer {

GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBufferWithDramSenders(
        *mesh_device, bank_to_receivers, size, buffer_type);
}
}  // namespace ttnn::global_circular_buffer
