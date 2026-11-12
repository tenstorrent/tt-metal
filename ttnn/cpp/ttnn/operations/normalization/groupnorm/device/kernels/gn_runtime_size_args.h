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
constexpr std::uint32_t NUM_ARGS = 8;
}  // namespace gn_size_args
