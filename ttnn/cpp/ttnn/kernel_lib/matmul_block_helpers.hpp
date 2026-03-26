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

// Default no-op post-compute functor.
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace matmul_block_config

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A × B with spill/reload.
 *
 * Performs matrix multiplication using hardware block-level matmul_block LLK with
 * sub-block indexing and automatic spill/reload for K-dimension blocking.
 *
 * Uses mm_block_init + matmul_block (the LLK function with hardware unroll) for
 * optimal performance. The spill/reload path uses _with_dt variants for correct
 * data format reconfiguration when in1_cb and interm_cb have different formats.
 *
 * NOTE: Unlike matmul_tile, this helper does NOT require compute_kernel_hw_startup()
 * to be called first. The mm_block_init() call inside the helper performs all
 * necessary hardware initialization.
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
 *   transpose  — If true, transpose B tiles before multiplication (default: false).
 *
 * ── PostComputeFn ────────────────────────────────────────────────────────
 *
 *   Optional functor called on each output sub-block after the last K-block's
 *   matmul, before tiles are packed. Receives out_subblock_num_tiles as argument.
 *   Tiles are in DST registers at indices 0..num_tiles-1. Use for fused SFPU
 *   activations (relu, gelu, etc.) on the final matmul output.
 *
 *   Example:
 *     struct ApplyRelu {
 *         ALWI void operator()(uint32_t num_tiles) const {
 *             for (uint32_t i = 0; i < num_tiles; i++) {
 *                 SFPU_OP_FUNC_ACTIVATION  // or relu_tile(i)
 *             }
 *         }
 *     };
 *     compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm,
 *         ..., ApplyRelu>(..., ApplyRelu{});
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
    matmul_block_config::InitUninitMode init_uninit_mode = matmul_block_config::InitUninitMode::InitAndUninit,
    matmul_block_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        matmul_block_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    bool transpose = false,
    typename PostComputeFn = matmul_block_config::NoPostCompute>
ALWI void matmul_block(
    matmul_block_config::In0BlockParams in0,
    matmul_block_config::In1BlockParams in1,
    uint32_t num_blocks,
    matmul_block_config::OutSubblockParams out,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
