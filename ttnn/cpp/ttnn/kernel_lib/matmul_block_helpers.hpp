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

// Block parameters for the A (in0) input matrix.
struct In0BlockParams {
    uint32_t block_w;             // Inner block dimension in tiles (K dimension block size)
    uint32_t num_subblocks;       // Number of sub-blocks along the M dimension
    uint32_t block_num_tiles;     // Total tiles per A block (= subblock_h * block_w * num_subblocks)
    uint32_t subblock_num_tiles;  // Tiles per A sub-block (= subblock_h * block_w)
};

// Block parameters for the B (in1) input matrix.
struct In1BlockParams {
    uint32_t num_subblocks;    // Number of sub-blocks along the N dimension
    uint32_t block_num_tiles;  // Total tiles per B block (= subblock_w * block_w * num_subblocks)
    uint32_t per_core_w;       // Tiles per B row (= subblock_w * num_subblocks)
};

// Output sub-block dimensions.
struct OutSubblockParams {
    uint32_t h;          // Output sub-block height in tiles
    uint32_t w;          // Output sub-block width in tiles
    uint32_t num_tiles;  // Tiles per output sub-block (= h * w)
};

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
 *   in0        — A-matrix block parameters (see In0BlockParams).
 *   in1        — B-matrix block parameters (see In1BlockParams).
 *   num_blocks — Number of blocks along the K dimension.
 *   out        — Output sub-block dimensions (see OutSubblockParams).
 *   batch      — Number of independent batch slices (default: 1).
 *
 * ── CB Sizing Requirements ─────────────────────────────────────────────────
 *
 *   in0_cb:    >= in0.block_num_tiles pages (full A block loaded at once)
 *   in1_cb:    >= in1.block_num_tiles pages (full B block loaded at once)
 *   out_cb:    >= total output tiles for reservation tracking
 *   interm_cb: >= out.num_tiles pages (one sub-block of partial results)
 *
 * ── Examples ───────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
 *   using namespace compute_kernel_lib::matmul_block_config;
 *
 *   // Sub-blocked matmul: 4x4 output divided into 2x2 sub-blocks, inner block width 2
 *   compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
 *       {.block_w = 2, .num_subblocks = 2, .block_num_tiles = 8, .subblock_num_tiles = 4},
 *       {.num_subblocks = 2, .block_num_tiles = 8, .per_core_w = 4},
 *       3,  // num_blocks
 *       {.h = 2, .w = 2, .num_tiles = 4},
 *       1); // batch
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
    matmul_block_config::In0BlockParams in0,
    matmul_block_config::In1BlockParams in1,
    uint32_t num_blocks,
    matmul_block_config::OutSubblockParams out,
    uint32_t batch = 1);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
