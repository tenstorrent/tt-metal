// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/dram_sender_global_circular_buffer.hpp"

#include <utility>

#include <tt-metalium/dram_subchannel.hpp>

namespace ttnn::dram_sender_global_circular_buffer {

namespace {

std::vector<std::pair<CoreCoord, CoreRangeSet>> build_mapping(
    IDevice* device, const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(bank_to_receivers.size());
    for (const auto& [bank_id, receivers] : bank_to_receivers) {
        uint32_t sub = tt::tt_metal::experimental::pick_unused_dram_subchannel(device, bank_id);
        mapping.emplace_back(CoreCoord{bank_id, sub}, receivers);
    }
    return mapping;
}

}  // namespace

DramSenderGlobalCircularBuffer create_dram_sender_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_mapping(device, bank_to_receivers);
    return tt::tt_metal::experimental::CreateDramSenderGlobalCircularBuffer(device, mapping, size, buffer_type);
}

DramSenderGlobalCircularBuffer create_dram_sender_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_mapping(mesh_device, bank_to_receivers);
    return tt::tt_metal::experimental::CreateDramSenderGlobalCircularBuffer(mesh_device, mapping, size, buffer_type);
}

}  // namespace ttnn::dram_sender_global_circular_buffer
