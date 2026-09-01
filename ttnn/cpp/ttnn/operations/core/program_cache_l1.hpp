// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::operations::core {

inline constexpr std::uint64_t kProgramL1UsagePercent = 95;

inline std::uint64_t available_program_l1_capacity(const tt::tt_metal::IDevice* device) {
    const auto lowest_occupied_l1 = device->lowest_occupied_compute_l1_address().value_or(device->l1_size_per_core());
    const auto cb_l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return lowest_occupied_l1 > cb_l1_base ? lowest_occupied_l1 - cb_l1_base : 0;
}

// Keep a conservative margin for non-CB L1 users. Callers should hash the
// selected program plan, not this allocator-dependent byte count.
inline std::uint64_t usable_program_l1_capacity(const tt::tt_metal::IDevice* device) {
    return available_program_l1_capacity(device) * kProgramL1UsagePercent / 100;
}

}  // namespace ttnn::operations::core
