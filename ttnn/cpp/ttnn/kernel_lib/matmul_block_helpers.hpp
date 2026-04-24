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

/**
 * Output pack-layout mode selected at compile time on the last K-block.
 *
 * SubblockMajor (default, legacy): sequential pack_tile_block per subblock;
 *   writer reads tiles in subblock-major order. Required by factories that
 *   emit subblock-major writer kernels.
 * RowMajor: absolute-offset pack_tile<true> into row-major positions within
 *   the M-row-group; reserve/push per row-group. Decouples subblock choice
 *   from output layout so factories can grow subblocks freely, and is the
 *   mode SDPA callers require for absolute-offset partial writes.
 *
 * When matmul_block feeds add_bias_bcast_rows, both must use the same
 * OutputLayout so the intermediate CB layout matches.
 */
enum class OutputLayout { SubblockMajor, RowMajor };

/**
 * Block-shape specification for matmul_block.
 *
 * Groups the dimensional params — subblock counts, subblock size, K-blocking,
 * batch — into one struct so callers pass intent instead of seven positional
 * integers. Optional strides (in1_per_core_w / out_row_width) stay on the
 * function signature because they're advanced layout overrides only a few
 * factories need.
 *
 * Usage:
 *   matmul_block<...>(
 *       in0_cb, in1_cb, out_cb, interm_cb,
 *       MatmulBlockShape::of(in0_sb, in1_sb, h, w, in0_block_w, num_k_blocks),
 *       ...);
 */
struct MatmulBlockShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_w;        // K per K-block in tiles (= A's "per_core_K_block").
    uint32_t num_k_blocks;       // Number of K-blocks along the K dimension.
    uint32_t batch = 1;          // Independent batch slices.

    static constexpr MatmulBlockShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_w,
        uint32_t num_k_blocks,
        uint32_t batch = 1) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_w, num_k_blocks, batch};
    }
};

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
 * output-pack strategies selected at compile time via layout:
 *
 *   layout=SubblockMajor (default): sequential pack_tile_block per subblock.
 *     Output lands in subblock order. Required by multicast writers that expect
 *     subblock-order tile stream.
 *   layout=RowMajor: absolute-offset pack_tile<true> at row-major positions,
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
 *   layout            OutputLayout: SubblockMajor (default) or RowMajor (see above).
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
 *   shape              MatmulBlockShape (see above) — subblock counts, subblock size,
 *                      K-blocking, batch. Build with MatmulBlockShape::of(...).
 *   post_compute       PostComputeFn instance (default: {}).
 *   pre_k_block        PreKBlockFn instance (default: {}).
 *   retain_in0         When true, skip popping in0 on the last K-block so the caller
 *                      retains the data (SDPA reuses Q across K chunks). (default: false)
 *   in1_per_core_w     Actual number of N-tiles in the in1 CB per K-block (= what NCRISC
 *                      pushes per block). Defaults to 0, meaning derive from
 *                      out_subblock_w * in1_num_subblocks. Pass the real value when the
 *                      factory pads per_core_N_compute above the actual in1 shard width
 *                      (e.g. matmul_multicore_reuse_mcast_dram_sharded), otherwise the
 *                      helper will wait/pop wrong tile counts and deadlock.
 *   out_row_width      N-tiles per row in the OUTPUT CB layout (row stride for row_major
 *                      pack). Defaults to 0, meaning reuse in1_per_core_w. For most factories
 *                      in1 read stride and output pack stride coincide. DRAM-sharded is
 *                      the exception: it reads in1 at per_core_N_in1_sender (unpadded shard
 *                      width) but packs output at per_core_N_compute (padded after subblock-
 *                      growth); those factories must pass the larger pack stride here.
 *
 * @example
 *   // Simple K=1 non-blocked matmul, defaults everywhere (SubblockMajor pack).
 *   // MatmulBlockShape::of arg order is (in0_sb, in1_sb, h, w, in0_block_w, num_k_blocks, batch).
 *   mm_block_init(cb_in0, cb_in1, cb_intermed0, false, out_subblock_w, out_subblock_h, in0_block_w);
 *   matmul_block<>(cb_in0, cb_in1, cb_out, cb_intermed0,
 *                  MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                                        out_subblock_h, out_subblock_w,
 *                                        in0_block_w, 1));
 *
 * @example
 *   // Row-major output + packer-L1 accumulation across K, no fused bias.
 *   // Template order: transpose, packer_l1_acc, pack_last_to_interm, pack_relu, layout.
 *   matmul_block<false, true, false, false, OutputLayout::RowMajor>(
 *       cb_in0, cb_in1, cb_out, cb_intermed0,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_k_blocks));
 *
 * @example
 *   // SDPA-style: row-major pack, retain_in0 to reuse Q across K chunks, masked post-compute.
 *   matmul_block<transpose, false, false, false, OutputLayout::RowMajor,
 *                OptionalMaskPostCompute>(
 *       in0_cb, in1_cb, out_cb, in0_cb,  // interm_cb unused when num_k_blocks==1
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             subblock_h, subblock_w, in0_block_w, num_blocks),
 *       OptionalMaskPostCompute{...},
 *       NoPreKBlock{},
 *       true);  // retain_in0
 *
 * @example
 *   // FUSE_BIAS path: last K-block packs to interm_cb so add_bias_bcast_rows reads it.
 *   // DRAM-sharded passes explicit in1_per_core_w (shard width) and
 *   // out_row_width (padded pack width).
 *   matmul_block<in1_transpose_tile, l1_acc, true, false,
 *                output_layout, PostFn, PreFn>(
 *       in0_cb, in1_cb, out_cb, mm_partials_cb,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_blocks_inner_dim),
 *       PostFn{}, PreFn{},
 *       false,            // retain_in0
 *       in1_block_w,      // in1_per_core_w  (DRAM-sharded shard width)
 *       out_block_w);     // out_row_width   (DRAM-sharded padded pack width)
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    OutputLayout layout = OutputLayout::SubblockMajor,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock>
ALWI void matmul_block(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    MatmulBlockShape shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    bool retain_in0 = false,
    uint32_t in1_per_core_w = 0,
    uint32_t out_row_width = 0);

/**
 * matmul_reduce_inplace: in-place reduce via matmul using a single-tile column identity.
 *
 * Consumes subblock_h×subblock_w tiles from the front of `in_out_cb`, computes
 *   DST = matmul(in_out_cb[0..subblock_h], in1_cb[0]) × block_kt accumulation
 * and packs back onto `in_out_cb` — repeated num_subblocks times to reduce the CB in
 * place. This pattern breaks the standard in0_cb != out_cb invariant that `matmul_block`
 * enforces, so it lives in a dedicated helper; SDPA uses this to fold partial-sum
 * results along M via a column-identity tile in in1_cb.
 *
 * The helper absorbs mm_block_init_short + reconfig_data_format + wait_front on both
 * CBs — the caller only needs to have produced the requisite tiles.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in_out_cb       CB serving as both input and output (in-place).
 *   in1_cb          CB with the single column-identity tile (kept fronted, not popped).
 *   num_subblocks   Number of subblock iterations (= rows / subblock_h).
 *   subblock_h      Subblock height in tiles (matmul rt_dim).
 *   subblock_w      Subblock width in tiles (matmul ct_dim; typically 1).
 *   block_kt        K dimension in tiles for each matmul call (typically 1 = subblock_w).
 *
 * @example
 *   // SDPA fold M partial-sums using a column-identity tile in in1_cb.
 *   // Before: cb_out_accum has (STATS_GRANULARITY * Wt) tiles;
 *   // After:  cb_out_accum has Wt tiles (one reduced row).
 *   // Args: (in_out_cb, in1_cb, num_subblocks, subblock_h, subblock_w=1, block_kt=1).
 *   matmul_reduce_inplace(cb_out_accum, cb_col_identity, Wt, STATS_GRANULARITY);
 */
ALWI void matmul_reduce_inplace(
    uint32_t in_out_cb,
    uint32_t in1_cb,
    uint32_t num_subblocks,
    uint32_t subblock_h,
    uint32_t subblock_w = 1,
    uint32_t block_kt = 1);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
