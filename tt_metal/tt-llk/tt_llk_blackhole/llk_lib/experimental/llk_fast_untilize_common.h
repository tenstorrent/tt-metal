// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"

namespace ckernel
{

constexpr std::uint32_t FAST_UNTILIZE_MAX_UNIT_DIM = 4;
constexpr std::uint32_t FAST_UNTILIZE_NUM_FACES    = 4;

// Shared math/pack DEST layout constants. One face is 16 DEST rows, and each
// tile contributes two 16-row face-pair strips: F0/F1 and F2/F3.
constexpr std::uint32_t FAST_UNTILIZE_PHASE_ROWS        = FACE_R_DIM;
constexpr std::uint32_t FAST_UNTILIZE_TILE_STRIDE_ROWS  = 2 * FAST_UNTILIZE_PHASE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_BLOCK_STRIDE_ROWS = FAST_UNTILIZE_MAX_UNIT_DIM * FAST_UNTILIZE_PHASE_ROWS;

// Math places all top face-pair rows first, followed by all bottom face-pair
// rows. The packer uses this same separation as the Z/W source stride.
constexpr std::uint32_t FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS  = 2 * FAST_UNTILIZE_BLOCK_STRIDE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_TOP_STRIP_ROW_OFFSET    = 0;
constexpr std::uint32_t FAST_UNTILIZE_BOTTOM_STRIP_ROW_OFFSET = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS;

// BH pack phase selection is a DEST-target remap, not a physical DEST row
// index. These offsets intentionally do not mirror the math row offsets above:
// offset 128 exposes the top strip, while offset 0 exposes the bottom strip.
constexpr std::uint32_t FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET    = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET = 0;

// Fast-untilize owns a private half-sync DEST region so each <=4-tile chunk can
// double-buffer independently of the ambient kernel sync mode.
constexpr DstSync FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE = DstSync::SyncHalf;

constexpr std::uint32_t fast_untilize_next_unit_dim(const std::uint32_t remaining_tiles)
{
    // Avoid a trailing unit_dim=1, which the fast pack/math path does not support.
    // A 5-tile tail is decomposed as 2 + 3.
    return (remaining_tiles > 5) ? FAST_UNTILIZE_MAX_UNIT_DIM : (remaining_tiles == 5) ? 2 : remaining_tiles;
}

inline std::uint32_t fast_untilize_decompose_row(const std::uint32_t ct_dim, std::uint32_t* const unit_dims)
{
    std::uint32_t idx = 0;
    for (std::uint32_t remaining = ct_dim; remaining > 0;)
    {
        const std::uint32_t unit_dim = fast_untilize_next_unit_dim(remaining);
        unit_dims[idx++]             = unit_dim;
        remaining -= unit_dim;
    }
    return idx;
}

} // namespace ckernel
