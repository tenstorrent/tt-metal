// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"

namespace ckernel
{

// The Blackhole fast-tilize MOP processes 2, 3 or 4 tiles per unit; there is no unit_dim 1.
constexpr std::uint32_t FAST_TILIZE_MIN_UNIT_DIM = 2;

/**
 * @brief Whether the Blackhole fast-tilize path supports a block of tiles of the given shape and width.
 *
 * The path is 32x32-tile only on every thread: the unpack MOP reads 32 tensor rows per chunk, math copies
 * them at the Tile32x32 DEST stride and pack replays 16 PACRs per tile with a full-tile L1 advance. A
 * block narrower than the smallest unit falls back as well. Any other block must take the plain tilize path.
 *
 * @param tile_r_dim: Tile height in datums of the operand being tilized.
 * @param tile_c_dim: Tile width in datums of the operand being tilized.
 * @param block: Block width in tiles.
 */
constexpr bool fast_tilize_block_supported(const std::uint32_t tile_r_dim, const std::uint32_t tile_c_dim, const std::uint32_t block)
{
    return block >= FAST_TILIZE_MIN_UNIT_DIM && tile_r_dim == TILE_R_DIM && tile_c_dim == TILE_C_DIM;
}

} // namespace ckernel
