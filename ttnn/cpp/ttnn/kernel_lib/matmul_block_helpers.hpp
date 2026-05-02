// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
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
 * Where the last K-block packs and what post-op it gets, picked at compile time.
 *
 * Out          (default) Last block packs to out_buf with no relu.
 * OutWithRelu  Last block packs to out_buf with PACK_RELU enabled.
 * Interm       Last block packs to interm_buf for a downstream phase (bias add /
 *              untilize) to consume. RELU on this path lives in the downstream
 *              phase, not the matmul.
 *
 * Replaces the previous (pack_last_to_interm, pack_relu) bool pair: the impossible
 * combination (Interm + Relu) is unrepresentable.
 */
enum class LastBlockTarget : uint8_t { Out, OutWithRelu, Interm };

namespace matmul_config {

/**
 * Init lifecycle for matmul_block.
 *
 * The helper owns matmul-state setup so callers don't have to pair every call site with a
 * matching mm_block_init. Mirrors the InitUninitMode convention used by untilize_helpers
 * for back-to-back invocations.
 *
 * Full          (default) Helper calls mm_block_init at the start. Use for the first matmul
 *               invocation after compute_kernel_hw_startup or after any intervening op
 *               (eltwise, reduce, untilize) that disturbed matmul LLK state.
 * Short         Helper calls mm_block_init_short — cheaper restore for chains of matmul
 *               calls where the previous helper already configured matmul state and only
 *               the per-call shape/cb might have changed.
 * None          Helper skips init entirely. Use only when the caller has just executed a
 *               compatible matmul_block in the same configuration and no other op has
 *               touched matmul state.
 */
enum class InitMode : uint8_t { Full, Short, None };

}  // namespace matmul_config

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
 *       in0_buf, in1_buf, out_buf, interm_buf,
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
    uint32_t batch = 1;          // Independent batch slices. Pass the actual batch count
                                 // when the caller has no per-batch work between matmuls
                                 // (matmul-only kernels): the helper's own batch loop runs
                                 // mm_block_init exactly once across all batches, which is
                                 // both faster and avoids the heterogeneous-tile-shape
                                 // re-init corruption fixed in commit 76e99730d2e. Keep
                                 // batch=1 and run the kernel's own batch loop ONLY when
                                 // per-batch phase work (bias add, untilize, mailbox sync)
                                 // must be interleaved between iterations.

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
 * Init handling: by default the helper calls mm_block_init() itself (init_mode=Full).
 * The caller's only init responsibility is one compute_kernel_hw_startup() at boot.
 * For back-to-back chains, init_mode=Short uses mm_block_init_short (cheap restore);
 * init_mode=None skips init entirely. See matmul_config::InitMode.
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
 *   last_block_target LastBlockTarget: Out (default), OutWithRelu, or Interm.
 *                     See LastBlockTarget docstring for the three valid pack/RELU
 *                     combinations.
 *   layout            OutputLayout: SubblockMajor (default) or RowMajor (see above).
 *   init_mode         matmul_config::InitMode: Full (default), Short, or None.
 *                     Controls whether the helper itself calls mm_block_init / _short.
 *   retain_in0        When true, skip popping in0 on the last K-block so the caller
 *                     retains the data (SDPA reuses Q across K chunks). (default: false)
 *   retain_in1        When true, skip popping in1 on the last K-block so the caller
 *                     retains the data (conv3d reuses weights across multiple matmul
 *                     invocations within an output block). (default: false)
 *   PostComputeFn     Functor called per output sub-block on the last K-block,
 *                     after matmul but before packing. Receives out_subblock_num_tiles.
 *   PreKBlockFn       Functor called at the start of each K-block iteration, before
 *                     input CB waits. Receives (block, num_k_blocks, is_last).
 *                     Use for per-K-block preprocessing such as in0_transpose.
 *   pin_interm_to_captured_base
 *                     Default false. When true, the helper captures interm_buf's
 *                     fifo_rd_ptr / fifo_wr_ptr at entry and resets them per K-block
 *                     (and once after the K-loop on the pack_last_to_interm path) to
 *                     keep interm_buf operating at a fixed L1 base across all blocks.
 *                     Required when interm_buf is allocated to alias the output buffer
 *                     in L1 (e.g. conv2d's `partials_cb_uses_output=true` path) — the
 *                     K-loop's natural fifo advance would otherwise wrap and overwrite
 *                     previously packed output. Pin pattern matches conv2d's original
 *                     manual K-loop:
 *                       pack_last_to_interm:        rd+wr reset for block < num-1
 *                       !pack_last_to_interm: rd reset for block < num-1; wr reset for
 *                                             block < num-2 (so the last reload still
 *                                             finds the second-to-last block's data at
 *                                             advanced wr_ptr).
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_buf, in1_buf   Input buffers for matrices A and B (CircularBuffer or
 *                      DataflowBuffer — pass an experimental::CircularBuffer
 *                      object on legacy CB-backed kernels, or an
 *                      experimental::DataflowBuffer object on Metal-2.0 / DFB
 *                      kernels).
 *   out_buf            Output buffer for the final result.
 *   interm_buf         Intermediate buffer for K-blocking spill/reload or
 *                      L1-ACC FIFO. When num_k_blocks == 1 it is never read;
 *                      pass any non-output buffer (typically the same buffer
 *                      type as the others).
 *   shape              MatmulBlockShape (see above) — subblock counts, subblock size,
 *                      K-blocking, batch. Build with MatmulBlockShape::of(...).
 *   post_compute       PostComputeFn instance (default: {}).
 *   pre_k_block        PreKBlockFn instance (default: {}).
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
 *   // Simple K=1 non-blocked matmul, defaults everywhere (SubblockMajor pack,
 *   // init_mode=Full). Caller constructs experimental::CircularBuffer (or
 *   // DataflowBuffer) once and passes the object; the helper wraps sync calls
 *   // (wait_front / pop_front / reserve_back / push_back) on it and issues
 *   // mm_block_init internally.
 *   experimental::CircularBuffer in0_buf(cb_in0);
 *   experimental::CircularBuffer in1_buf(cb_in1);
 *   experimental::CircularBuffer out_buf(cb_out);
 *   experimental::CircularBuffer interm_buf(cb_intermed0);
 *   matmul_block<>(in0_buf, in1_buf, out_buf, interm_buf,
 *                  MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                                        out_subblock_h, out_subblock_w,
 *                                        in0_block_w, 1));
 *
 * @example
 *   // Row-major output + packer-L1 accumulation across K, no fused bias.
 *   // Template order: transpose, packer_l1_acc, last_block_target, layout.
 *   // Buf is deduced from the buffer-object arguments.
 *   matmul_block<false, true, LastBlockTarget::Out, OutputLayout::RowMajor>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_k_blocks));
 *
 * @example
 *   // SDPA-style: row-major pack, retain_in0 to reuse Q across K chunks, masked post-compute.
 *   // The SDPA-side wrapper does its own [mm_block_init_short, reconfig_data_format]
 *   // pair externally (for ordering parity with matmul_reduce_inplace.inl and
 *   // OptionalMaskPostCompute), so the helper is invoked with init_mode=None.
 *   // Template slot order: transpose, packer_l1_acc, last_block_target, layout,
 *   // init_mode, retain_in0, retain_in1, PostComputeFn.
 *   matmul_block<transpose, false, LastBlockTarget::Out, OutputLayout::RowMajor,
 *                matmul_config::InitMode::None, true, false,
 *                OptionalMaskPostCompute>(
 *       in0_buf, in1_buf, out_buf, in0_buf,  // interm unused when num_k_blocks==1
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             subblock_h, subblock_w, in0_block_w, num_blocks),
 *       OptionalMaskPostCompute{...},
 *       NoPreKBlock{});
 *
 * @example
 *   // FUSE_BIAS path: last K-block packs to interm_buf so add_bias_bcast_rows reads it.
 *   // DRAM-sharded passes explicit in1_per_core_w (shard width) and
 *   // out_row_width (padded pack width).
 *   matmul_block<in1_transpose_tile, l1_acc, LastBlockTarget::Interm,
 *                output_layout, matmul_config::InitMode::Full, false, false,
 *                PostFn, PreFn>(
 *       in0_buf, in1_buf, out_buf, mm_partials_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_blocks_inner_dim),
 *       PostFn{}, PreFn{},
 *       in1_block_w,      // in1_per_core_w  (DRAM-sharded shard width)
 *       out_block_w);     // out_row_width   (DRAM-sharded padded pack width)
 *
 * @example
 *   // conv2d pattern: per-K-block tilize (PreKBlockFn) + interm_buf pinned to a captured
 *   // base across the K-loop because matmul_partials_cb is allocated to alias out_cb in
 *   // L1 (partials_cb_uses_output=true). init_mode=None because the kernel-entry
 *   // mm_block_init covers initial state and ConvTilizePreKBlock issues
 *   // mm_block_init_short_with_both_dt after each tilize. Template slot order: transpose,
 *   // packer_l1_acc, last_block_target, layout, init_mode, retain_in0, retain_in1,
 *   // PostComputeFn, PreKBlockFn, pin_interm_to_captured_base.
 *   matmul_block<false, packer_l1_acc, LastBlockTarget::Interm,
 *                OutputLayout::SubblockMajor, matmul_config::InitMode::None,
 *                false, false, ConvSFPUPostCompute, ConvTilizePreKBlock,
 *                true>(  // pin_interm_to_captured_base
 *       cb_mm_in0, cb_in1, cb_matmul_partials, cb_matmul_partials,  // out==interm
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, in0_num_blocks_w),
 *       ConvSFPUPostCompute{}, conv_pre_k_block);
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    LastBlockTarget last_block_target = LastBlockTarget::Out,
    OutputLayout layout = OutputLayout::SubblockMajor,
    matmul_config::InitMode init_mode = matmul_config::InitMode::Full,
    bool retain_in0 = false,
    bool retain_in1 = false,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    bool pin_interm_to_captured_base = false,
    typename Buf = ::experimental::CircularBuffer>
ALWI void matmul_block(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    MatmulBlockShape shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
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
 *   in_out_buf      Buffer serving as both input and output (in-place).
 *   in1_buf         Buffer with the single column-identity tile (kept
 *                   fronted, not popped).
 *   num_subblocks   Number of subblock iterations (= rows / subblock_h).
 *   subblock_h      Subblock height in tiles (matmul rt_dim).
 *   subblock_w      Subblock width in tiles (matmul ct_dim; typically 1).
 *   block_kt        K dimension in tiles for each matmul call (typically 1 = subblock_w).
 *
 * @example
 *   // SDPA fold M partial-sums using a column-identity tile in in1_buf.
 *   // Before: out_accum_buf has (STATS_GRANULARITY * Wt) tiles;
 *   // After:  out_accum_buf has Wt tiles (one reduced row).
 *   matmul_reduce_inplace(out_accum_buf, col_identity_buf, Wt, STATS_GRANULARITY);
 */
template <typename Buf = ::experimental::CircularBuffer>
ALWI void matmul_reduce_inplace(
    Buf& in_out_buf,
    Buf& in1_buf,
    uint32_t num_subblocks,
    uint32_t subblock_h,
    uint32_t subblock_w = 1,
    uint32_t block_kt = 1);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
