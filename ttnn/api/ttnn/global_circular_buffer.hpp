// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/global_circular_buffer.hpp>
// --- NUKED OPS: matmul removed; matmul-1d GCB helper dropped ---
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

// Single Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Multi Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// DRAM-sender variant: senders are programmable DRAM cores identified by DRAM bank id.
// The returned GlobalCircularBuffer is the same type as the worker variant; the sender
// domain is queryable via tt::tt_metal::experimental::sender_core_type(gcb).
GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::global_circular_buffer
