// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/transpose_wh.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

/**
 * transpose_tile_block: WH-transpose a block of tiles from one CB to another.
 *
 * Reads from `in_transpose_cb`, applies transpose_wh_tile to each tile, packs
 * into `in_cb`. Processes in chunks of `block_size` tiles (default 4, matches
 * DST-register capacity across dst_sync modes and data formats) with a tail
 * loop for the remainder.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   in_block_num_tiles  Total tiles in the block.
 *   block_size          Tiles per DST batch (default 4).
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in_transpose_cb  CB to read original tiles from.
 *   in_cb            CB to write transposed tiles to.
 */
template <uint32_t in_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in_transpose_cb, uint32_t in_cb);

/**
 * TransposePreKBlock: PreKBlockFn functor that transposes in0 before each K-block
 * iteration of a matmul_block call.
 *
 * Reconfigures data formats, runs transpose_tile_block to populate the transposed
 * input CB, then re-inits matmul and restores pack format. Designed to be passed
 * as the PreKBlockFn template argument to compute_kernel_lib::matmul_block when
 * the matmul needs its in0 WH-transposed on the fly.
 *
 * All parameters are compile-time template args so the functor has no state.
 */
template <
    uint32_t in0_block_num_tiles,
    uint32_t in0_transpose_cb_id,
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w,
    uint32_t mm_partials_cb_id>
struct TransposePreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const;
};

}  // namespace compute_kernel_lib

#include "transpose_block_helpers.inl"
