// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::operations::core {

inline constexpr std::uint64_t kProgramL1UsagePercent = 95;

constexpr std::uint64_t available_program_l1_capacity(
    std::uint64_t lowest_occupied_l1, std::uint64_t cb_l1_base, std::uint64_t bank_capacity) {
    // Shared by normalisation, reductions, matmul and data movement. An occupied
    // address below the CB base must not wrap to a large unsigned capacity.
    const auto unoccupied_span = lowest_occupied_l1 > cb_l1_base ? lowest_occupied_l1 - cb_l1_base : 0;
    return std::min(unoccupied_span, bank_capacity);
}

inline std::uint64_t available_program_l1_capacity(const tt::tt_metal::IDevice* device) {
    const auto lowest_occupied_l1 = device->lowest_occupied_compute_l1_address().value_or(device->l1_size_per_core());
    const auto cb_l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    // The occupancy query ignores L1_SMALL. The ordinary bank's byte capacity
    // already excludes both the CB base and the reserved L1_SMALL region, even
    // when that region has no live tensors.
    const auto bank_capacity = device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
    return available_program_l1_capacity(lowest_occupied_l1, cb_l1_base, bank_capacity);
}

// Keep a conservative margin for non-CB L1 users. Callers should hash the
// selected program plan, not this allocator-dependent byte count.
inline std::uint64_t usable_program_l1_capacity(const tt::tt_metal::IDevice* device) {
    return available_program_l1_capacity(device) * kProgramL1UsagePercent / 100;
}

}  // namespace ttnn::operations::core
