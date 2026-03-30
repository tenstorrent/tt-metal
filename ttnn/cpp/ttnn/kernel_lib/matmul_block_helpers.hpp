// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

namespace matmul_block_config {

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
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   in0_cb     — Input CB for matrix A (0–31).
 *   in1_cb     — Input CB for matrix B (0–31).
 *   out_cb     — Output CB for final result C (0–31).
 *   interm_cb  — Intermediate CB for partial result spill/reload (0–31).
 *                Only used when num_k_blocks > 1. Must differ from out_cb.
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
 *         false, ApplyRelu>(..., ApplyRelu{});
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   block_w            — Inner block dimension in tiles (K-dimension block size).
 *   in0_num_subblocks  — Number of sub-blocks along the M dimension.
 *   in1_num_subblocks  — Number of sub-blocks along the N dimension.
 *   num_k_blocks       — Number of blocks along the K dimension.
 *   out_subblock_h     — Output sub-block height in tiles.
 *   out_subblock_w     — Output sub-block width in tiles.
 *   batch              — Number of independent batch slices (default: 1).
 *
 * ── Derived Quantities (computed internally) ────────────────────────────────
 *
 *   out_num_tiles          = out_subblock_h * out_subblock_w
 *   in0_subblock_num_tiles = out_subblock_h * block_w
 *   in0_block_num_tiles    = in0_subblock_num_tiles * in0_num_subblocks
 *   in1_per_core_w         = out_subblock_w * in1_num_subblocks
 *   in1_block_num_tiles    = out_subblock_w * block_w * in1_num_subblocks
 *
 * ── CB Sizing Requirements ──────────────────────────────────────────────────
 *
 *   in0_cb:    >= in0_block_num_tiles pages (full A block)
 *   in1_cb:    >= in1_block_num_tiles pages (full B block)
 *   out_cb:    >= total output tiles (for reservation tracking)
 *   interm_cb: >= out_num_tiles pages (partial result spill, only when num_k_blocks > 1)
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    bool transpose = false,
    typename PostComputeFn = matmul_block_config::NoPostCompute>
ALWI void matmul_block(
    uint32_t block_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
