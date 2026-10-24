// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Per-core work split from one runtime length word. Host factories and device kernels derive every
// length-dependent count from these, so a sender and its receiver can never disagree on the split.
namespace ttnn::runtime_length {

constexpr uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

constexpr uint32_t tiles_per_core(uint32_t total_tiles, uint32_t num_cores) { return div_up(total_tiles, num_cores); }

// First tile of core `core_idx`; equals `total` (empty range) for cores past the end at short lengths.
constexpr uint32_t core_start(uint32_t core_idx, uint32_t per_core, uint32_t total) {
    return min_u32(core_idx * per_core, total);
}

constexpr uint32_t core_end(uint32_t core_idx, uint32_t per_core, uint32_t total) {
    return min_u32((core_idx + 1) * per_core, total);
}

constexpr uint32_t blocks_per_core(uint32_t per_core_tiles, uint32_t block_tiles) {
    return div_up(per_core_tiles, block_tiles);
}

// Tile count after every core's share is rounded up to whole blocks.
constexpr uint32_t padded_tiles(uint32_t total_tiles, uint32_t num_cores, uint32_t block_tiles) {
    return blocks_per_core(tiles_per_core(total_tiles, num_cores), block_tiles) * block_tiles * num_cores;
}

constexpr uint32_t rounds(uint32_t rows, uint32_t workers) { return div_up(rows, workers); }

// Balanced split in split_work_to_cores order: the first `total % num_cores` cores take one extra unit.
constexpr uint32_t balanced_start(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    return core_idx * (total / num_cores) + min_u32(core_idx, total % num_cores);
}

constexpr uint32_t balanced_end(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    return balanced_start(core_idx + 1, total, num_cores);
}

}  // namespace ttnn::runtime_length
