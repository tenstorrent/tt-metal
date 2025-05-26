// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <memory>
#include <tt-metalium/global_circular_buffer.hpp>

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

}  // namespace ttnn::global_circular_buffer
