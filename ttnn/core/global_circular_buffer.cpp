// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <memory>
#include <utility>
#include <tt-metalium/dram_subchannel.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::global_circular_buffer {

namespace {

// Build a (DRAM CoreCoord, receivers) mapping from a (bank_id, receivers) mapping by
// picking an unused subchannel for each bank id.
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
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
    IDevice* device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_dram_sender_mapping(device, bank_to_receivers);
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, mapping, size, buffer_type, tt::tt_metal::experimental::SenderCoreType::Dram);
}

GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_dram_sender_mapping(mesh_device, bank_to_receivers);
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        mesh_device, mapping, size, buffer_type, tt::tt_metal::experimental::SenderCoreType::Dram);
}

}  // namespace ttnn::global_circular_buffer
