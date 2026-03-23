// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"

namespace compute_kernel_lib {

namespace matmul_block_config {

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure
};

// Controls whether mm_init is called.
// Note: there is no mm_uninit in the LLK API, so UninitOnly and Neither are both no-ops.
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

}  // namespace matmul_block_config

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A × B with spill/reload.
 *
 * Performs matrix multiplication using sub-blocks that fit within DST registers.
 * When the inner dimension (K) is split across multiple blocks (num_blocks > 1),
 * partial results are spilled to an intermediate CB (interm_cb) and reloaded for
 * accumulation, enabling matmul of larger matrices than DST can hold at once.
 *
 * This wraps mm_init + matmul_tiles with sub-block indexing, matching the pattern
 * used in the bmm_large_block_zm programming example. Input tiles are consumed in
 * full blocks (all in0_block_num_tiles and in1_block_num_tiles at once) rather than
 * one-at-a-time as in matmul_tile.
 *
 * NOTE: Unlike matmul_tile, this helper does NOT require compute_kernel_hw_startup()
 * to be called first. The mm_init() call inside the helper performs all necessary
 * hardware initialization. Calling compute_kernel_hw_startup() before this helper
 * may cause incorrect results due to conflicting hardware configuration.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   in0_cb     — Input CB for matrix A (0–31).
 *   in1_cb     — Input CB for matrix B (0–31).
 *   out_cb     — Output CB for final result C (0–31).
 *   interm_cb  — Intermediate CB for partial result spill/reload (0–31).
 *                Only used when num_blocks > 1. Must differ from out_cb.
 *                The out_cb and interm_cb should share memory (overlapping address
 *                space) to avoid wasting L1 — the output only needs space once the
 *                final block is ready.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_block_w           — Inner block dimension in tiles (K dimension block size).
 *   in0_num_subblocks     — Number of sub-blocks along the M dimension.
 *   in0_block_num_tiles   — Total tiles per A block (= out_subblock_h * in0_block_w * in0_num_subblocks).
 *   in0_subblock_num_tiles — Tiles per A sub-block (= out_subblock_h * in0_block_w).
 *   in1_num_subblocks     — Number of sub-blocks along the N dimension.
 *   in1_block_num_tiles   — Total tiles per B block (= out_subblock_w * in0_block_w * in1_num_subblocks).
 *   in1_per_core_w        — Tiles per B row (= out_subblock_w * in1_num_subblocks).
 *   num_blocks            — Number of blocks along the K dimension.
 *   out_subblock_h        — Output sub-block height in tiles.
 *   out_subblock_w        — Output sub-block width in tiles.
 *   out_subblock_num_tiles — Tiles per output sub-block (= out_subblock_h * out_subblock_w).
 *   batch                 — Number of independent batch slices (default: 1).
 *
 * ── CB Sizing Requirements ─────────────────────────────────────────────────
 *
 *   in0_cb:    >= in0_block_num_tiles pages (full A block loaded at once)
 *   in1_cb:    >= in1_block_num_tiles pages (full B block loaded at once)
 *   out_cb:    >= total output tiles for reservation tracking
 *   interm_cb: >= out_subblock_num_tiles pages (one sub-block of partial results)
 *
 * ── Examples ───────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
 *
 *   // Sub-blocked matmul: 4x4 output divided into 2x2 sub-blocks, inner block width 2
 *   compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
 *       2,           // in0_block_w
 *       2, 8, 4,     // in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles
 *       2, 8, 4,     // in1_num_subblocks, in1_block_num_tiles, in1_per_core_w
 *       3,           // num_blocks
 *       2, 2, 4,     // out_subblock_h, out_subblock_w, out_subblock_num_tiles
 *       1);          // batch
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    matmul_block_config::InitUninitMode init_uninit_mode = matmul_block_config::InitUninitMode::InitAndUninit,
    matmul_block_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        matmul_block_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>
ALWI void matmul_block(
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t batch = 1);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
