// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// With GN_RUNTIME_SIZE_ARGS defined, the welford GroupNorm kernels read the values that depend on the
// input's spatial size from common runtime args instead of named compile-time args, so a new input
// size reuses the compiled kernels. The program factory writes the same list to every kernel.
namespace gn_size_args {
constexpr std::uint32_t PER_CORE_M = 0;           // per-core block height in tiles
constexpr std::uint32_t BLOCK_H = 1;              // == PER_CORE_M for one batch per core
constexpr std::uint32_t NUM_OUT_BLOCKS = 2;       // L1 split of the block height
constexpr std::uint32_t NUM_TILES_PER_BATCH = 3;  // per-core tile count of one batch
constexpr std::uint32_t NUM_MCAST_CORES = 4;      // cores per mcast (reduction) group
constexpr std::uint32_t TWO_PASS_RECIPROCAL = 5;  // fp32 bits of the two-pass mean reciprocal
constexpr std::uint32_t PER_CORE_MN = 6;          // per-core tile count, used with TILIZE_IN
constexpr std::uint32_t NUM_ROWS_PER_GROUP = 7;   // per-core rows (not tiles) of each group
// When the spatial extent is not whole tiles, its last tile row holds padding rows. The readers zero
// them, so the statistics see zeros, and the sender corrects the statistics for those zeros.
constexpr std::uint32_t PAD_ROW_FIRST_PAGE = 8;  // first page of the tile row with padding rows
constexpr std::uint32_t PAD_ROW_VALID_ROWS = 9;  // valid rows in that tile row, 32 when there is none
constexpr std::uint32_t PAD_STATS_SCALE = 10;    // fp32 bits of padded / logical spatial extent
constexpr std::uint32_t NUM_ARGS = 11;
}  // namespace gn_size_args
