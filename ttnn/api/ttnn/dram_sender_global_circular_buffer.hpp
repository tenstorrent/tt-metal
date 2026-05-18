// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/dram_sender_global_circular_buffer.hpp>

#include "ttnn/types.hpp"

namespace ttnn::dram_sender_global_circular_buffer {

// Helper: takes a (bank_id, receivers) mapping, picks the unused DRAM subchannel for each
// bank, and constructs a DramSenderGlobalCircularBuffer with sender coords at
// CoreCoord{bank_id, unused_subchannel}. Receiver sets across senders must be disjoint.
DramSenderGlobalCircularBuffer create_dram_sender_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

DramSenderGlobalCircularBuffer create_dram_sender_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::dram_sender_global_circular_buffer
