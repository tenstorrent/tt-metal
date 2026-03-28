// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"

namespace compute_kernel_lib {

namespace matmul_block_fused_bias_config {

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure
};

// Controls whether mm_block_init is called.
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

// Default no-op post-compute functor (applied after bias add, before packing).
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace matmul_block_fused_bias_config

/**
 * matmul_block_fused_bias: sub-blocked tiled matmul with fused bias addition.
 *
 * Performs C = A × B + bias, where bias is broadcast along rows (each row of
 * output tiles gets the same bias tile added). Optionally applies an SFPU
 * activation function after the bias add.
 *
 * This helper implements the common production pattern from
 * bmm_large_block_zm_fused_bias_activation.cpp:
 *   1. Sub-blocked matmul with spill/reload (same as matmul_block helper)
 *   2. Bias addition with row broadcast (add_bcast_rows)
 *   3. Optional SFPU activation via PostComputeFn
 *
 * The matmul phase packs results to interm_cb. The bias phase reads from
 * interm_cb, adds bias from bias_cb with row broadcast, applies optional
 * PostComputeFn, and packs final results to out_cb.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   in0_cb     — Input CB for matrix A (0–31).
 *   in1_cb     — Input CB for matrix B (0–31).
 *   out_cb     — Output CB for final result C + bias (0–31).
 *   interm_cb  — Intermediate CB for matmul partial results (0–31).
 *                Used for K-blocking spill/reload AND as the staging buffer
 *                between the matmul and bias phases.
 *   bias_cb    — Input CB for bias vector (0–31). Must contain per_core_w
 *                (= subblock_w * in1_num_subblocks) bias tiles, one per
 *                output column tile. Bias tiles persist in the CB across
 *                batches and M-blocks.
 *   transpose  — If true, transpose B tiles before multiplication (default: false).
 *
 * ── PostComputeFn ──────────────────────────────────────────────────────────
 *
 *   Optional functor called on each output sub-block after bias addition,
 *   before tiles are packed to out_cb. Receives out_subblock_num_tiles as
 *   argument. Tiles are in DST registers at indices 0..num_tiles-1.
 *   Use for fused SFPU activations (relu, gelu, etc.).
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0        — A-matrix block parameters (see In0BlockParams).
 *   in1        — B-matrix block parameters (see In1BlockParams).
 *   num_blocks — Number of blocks along the K dimension.
 *   out        — Output sub-block dimensions (see OutSubblockParams).
 *   batch      — Number of independent batch slices (default: 1).
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    uint32_t bias_cb,
    matmul_block_fused_bias_config::InitUninitMode init_uninit_mode =
        matmul_block_fused_bias_config::InitUninitMode::InitAndUninit,
    matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    bool transpose = false,
    typename PostComputeFn = matmul_block_fused_bias_config::NoPostCompute>
ALWI void matmul_block_fused_bias(
    matmul_block_fused_bias_config::In0BlockParams in0,
    matmul_block_fused_bias_config::In1BlockParams in1,
    uint32_t num_blocks,
    matmul_block_fused_bias_config::OutSubblockParams out,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib

#include "matmul_block_fused_bias_helpers.inl"
