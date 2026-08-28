// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::operations::core {

// L1-dependent program layouts must use this same conservative capacity in both
// their cache key and program factory. This avoids creating a cache entry for
// every allocator address while ensuring a cached layout fits throughout its bucket.
inline constexpr std::uint64_t PROGRAM_CACHE_L1_CAPACITY_GRANULARITY = 16 * 1024;

inline std::uint64_t program_cache_l1_capacity(const tt::tt_metal::IDevice* device) {
    const auto lowest_occupied_l1 = device->lowest_occupied_compute_l1_address().value_or(device->l1_size_per_core());
    const auto cb_l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const std::uint64_t available_l1 = lowest_occupied_l1 > cb_l1_base ? lowest_occupied_l1 - cb_l1_base : 0;
    const std::uint64_t usable_l1 = available_l1 * 95 / 100;
    return usable_l1 / PROGRAM_CACHE_L1_CAPACITY_GRANULARITY * PROGRAM_CACHE_L1_CAPACITY_GRANULARITY;
}

}  // namespace ttnn::operations::core
