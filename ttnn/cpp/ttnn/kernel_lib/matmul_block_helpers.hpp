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

// Default no-op post-compute functor.
// Called per output sub-block on the last K-block, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// Default no-op pre-K-block functor.
// Called at the start of each K-block iteration, before input CB waits.
// Receives (block_index, num_k_blocks, is_last_block).
// Use for per-K-block preprocessing (e.g., in0_transpose, global CB pointer manipulation).
struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A × B with K-blocking.
 *
 * One helper serving both standard matmul (non-multicast bmm) and SDPA. Supports two
 * output-pack strategies selected at compile time via row_major_output:
 *
 *   row_major_output=false (default): sequential pack_tile_block per subblock.
 *     Output lands in subblock order. Required by multicast writers that expect
 *     subblock-order tile stream.
 *   row_major_output=true: absolute-offset pack_tile<true> at row-major positions,
 *     reserve/push per M-row-group. Decouples subblock choice from output layout so
 *     the factory can pick larger subblocks (the main SDPA perf path). Also the
 *     mode SDPA callers require for absolute-offset partial writes across K chunks.
 *
 * K-accumulation also selectable at compile time:
 *   packer_l1_acc=false: Software spill/reload via interm_cb
 *   packer_l1_acc=true:  Hardware L1 accumulation via packer (no spill/reload)
 *
 * PREREQUISITE: Caller must call mm_block_init() before invoking this helper.
 *
 * SKIP_COMPUTE: When this macro is defined by the calling TU (microbenchmark path),
 * the inner ckernel::matmul_block() call is omitted. All other pipeline work (waits,
 * reloads, packs, L1_ACC toggles) still runs so the harness measures non-compute
 * overhead. Handled inside this helper — caller does nothing special.
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release) for correct
 * MATH-PACK pipelining.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   transpose         If true, transpose B tiles before multiplication (default: false).
 *   packer_l1_acc     Enable packer L1 accumulation instead of software spill/reload.
 *   pack_last_to_interm  If true, last K-block packs to interm_cb instead of out_cb.
 *                     Use when a post-processing phase (bias add, untilize) reads
 *                     from interm_cb.
 *   pack_relu         Enable PACK_RELU on the last K-block when !pack_last_to_interm.
 *   row_major_output  Enable absolute-offset packing on the last K-block (see above).
 *   PostComputeFn     Functor called per output sub-block on the last K-block,
 *                     after matmul but before packing. Receives out_subblock_num_tiles.
 *   PreKBlockFn       Functor called at the start of each K-block iteration, before
 *                     input CB waits. Receives (block, num_k_blocks, is_last).
 *                     Use for per-K-block preprocessing such as in0_transpose.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_cb, in1_cb     Input CBs for matrices A and B.
 *   out_cb             Output CB for the final result.
 *   interm_cb          Intermediate CB for K-blocking spill/reload or L1-ACC FIFO.
 *                      When num_k_blocks == 1 it is never read; pass any non-output CB.
 *   block_w            Inner block dimension in tiles (K-dimension block size).
 *   in0_num_subblocks  Number of sub-blocks along the M dimension.
 *   in1_num_subblocks  Number of sub-blocks along the N dimension.
 *   num_k_blocks       Number of blocks along the K dimension.
 *   out_subblock_h     Output sub-block height in tiles.
 *   out_subblock_w     Output sub-block width in tiles.
 *   batch              Number of independent batch slices (default: 1).
 *   post_compute       PostComputeFn instance (default: {}).
 *   pre_k_block        PreKBlockFn instance (default: {}).
 *   retain_in0         When true, skip popping in0 on the last K-block so the caller
 *                      retains the data (SDPA reuses Q across K chunks). (default: false)
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    bool row_major_output = false,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock>
ALWI void matmul_block(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    uint32_t block_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t batch = 1,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    bool retain_in0 = false);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
