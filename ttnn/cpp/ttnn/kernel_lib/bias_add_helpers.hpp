// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // OutputCBLayout
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {

/**
 * Bias-operand layout in bias_cb selected at compile time.
 *
 * RowBroadcast (default): one bias tile per output column, broadcast across
 *   all M rows of the sub-block via add_tiles_bcast_rows. Logical bias shape
 *   [1, N] / [..., 1, N].
 * Elementwise: bias has multiple M rows matching the output sub-block,
 *   added element-wise via add_tiles. Required when the bias spans multiple
 *   tile-rows rather than a single broadcast row.
 */
enum class BiasBroadcast { RowBroadcast, Elementwise };

/**
 * Subblock-grid and row-stride spec for add_bias_bcast_rows. Mirrors MatmulBlockShape's
 * first four args so a bias call under a matmul_block call reuses the same subblock values
 * (bias has no K dimension, so the K-blocking args are absent). Build with BiasAddShape::of(...).
 *
 * out_row_width is consulted only for tile_order == TileRowMajor; 0 (default) derives it
 * from out_subblock_w * in1_num_subblocks.
 */
struct BiasAddShape {
    uint32_t in0_num_subblocks;  // Subblock count along M.
    uint32_t in1_num_subblocks;  // Subblock count along N.
    uint32_t out_subblock_h;     // Subblock height in tiles.
    uint32_t out_subblock_w;     // Subblock width in tiles.
    uint32_t out_row_width = 0;  // Row stride in tiles for TileRowMajor pack; ignored for SubblockMajor.

    static constexpr BiasAddShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t out_row_width = 0) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, out_row_width};
    }
};

namespace bias_add_config {

// Default no-op post-bias functor.
// Called per output sub-block after bias addition, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
// Use for fused SFPU activation after bias (e.g., gelu, silu).
struct NoPostBias {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace bias_add_config

/**
 * add_bias_bcast_rows: bias addition on matmul output.
 *
 * Required includes:
 *   #include "api/compute/compute_kernel_hw_startup.h"  // for compute_kernel_hw_startup()
 *   #include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
 *
 * Reads matmul output sub-blocks from partials_cb, adds bias (either row-broadcast
 * or elementwise), optionally applies a post-bias operation (e.g., SFPU activation),
 * and packs the result to out_cb.
 *
 * The `broadcast` template param selects the add mode:
 *   - BiasBroadcast::RowBroadcast (default): one bias tile per output column, broadcast
 *                      across all M rows of the sub-block. Bias shape [1, N] / [..., 1, N].
 *   - BiasBroadcast::Elementwise: bias has multiple M rows matching the output sub-block;
 *                      required when the bias spans multiple tile-rows.
 *
 * This is a dedicated helper rather than a generic broadcast-add because it (1) consumes
 * SubblockMajor interm tiles in subblock order (a flat row-major add would need
 * subblock-aware indexing), (2) fuses SFPU activation on the PACKER thread, and (3)
 * supports a bias walk-offset (bias_offset). The add itself is order-independent.
 *
 * Composes with matmul_block by reading from the same interm_cb that matmul_block
 * packed to (when pack_last_to_interm=true). The `tile_order` template must match
 * the upstream matmul_block's layout so the intermediate CB is consumed in the right
 * order.
 *
 * CB flow (layout=SubblockMajor): partials_cb (wait+pop per subblock) + bias_cb
 *          (caller owns wait/pop) --> out_cb (reserve+push per subblock).
 * CB flow (layout=TileRowMajor):    partials_cb (wait+pop per M-row-group) + bias_cb
 *          (caller owns wait/pop) --> out_cb (reserve+push per M-row-group).
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release).
 *
 * PREREQUISITE: Caller handles PACK_RELU configuration, pack format reconfig,
 * L1_ACC disable, AND bias CB wait/pop lifecycle. Bias CB wait lifecycle depends
 * on caller's loop structure (reader often only pushes bias once across multiple
 * bh/batch iterations), so the helper does not touch bias_cb outside of reading
 * tiles for the broadcast add.
 *
 * ── Template parameters ──────────────────────────────────────────────────────
 *
 *   broadcast      BiasBroadcast::RowBroadcast (default, add_tiles_bcast_rows) or
 *                  Elementwise (add_tiles).
 *   tile_order     SubblockMajor (default) or TileRowMajor — MUST match the upstream
 *                  matmul_block layout so the interm tile order matches. See OutputCBLayout.
 *   PostBiasFn     per-subblock hook after bias add, before pack, on the MATH thread (default
 *                  NoPostBias). For fused activation prefer Activation (packer-thread, no
 *                  math-thread DST pressure).
 *   Activation     fuse an SFPU activation on the PACKER thread at the post-bias pack stage
 *                  (default none); replaces tile_regs_wait. Mirrors matmul_block's Activation
 *                  slot — this is where activation belongs when a matmul fed into bias. Build
 *                  from the sfpu_activation_helpers.hpp aliases (or ActivationOp<...> for
 *                  host-driven kinds); ActivationInitHelper::init() is the caller's boot-time job.
 *   in_place       TileRowMajor only (asserted): read-modify-write the bias result back into the
 *                  SAME partials CB it read from (partials_buf and out_buf must be the same CB),
 *                  instead of into a distinct out_buf. Mirrors the matmul_block in-place pack model:
 *                  the caller has done ONE reserve/push over the whole output block (matmul_block's
 *                  packs_in_place), so the fronted block already occupies its L1 region and the
 *                  helper reuses it — NO reserve_back, NO push_back, NO pop_front. This removes the
 *                  reserve-before-pop circular wait entirely (there is no reserve), so a dedicated
 *                  staging CB is unnecessary; a downstream in-kernel consumer (e.g. untilize) reads
 *                  the biased block from partials_buf and owns its pop.
 *                  PRECONDITION: partials CB is a ONE-output-block region whose fifo_wr_ptr has
 *                  wrapped back to the block base after the producer's push_back (guaranteed for a
 *                  one-block CB — cb_push_back wraps fifo_wr_ptr to base when it hits fifo_limit),
 *                  so pack_tile<out_of_order> lands each tile in place at base + tile_index.
 *
 * ── Runtime parameters ───────────────────────────────────────────────────────
 *
 *   partials_buf  matmul output (= interm buffer from matmul_block).
 *   bias_buf      bias tiles. RowBroadcast: one tile per output column. Elementwise: multiple
 *                 M rows per column.
 *   out_buf       biased result.
 *   shape         BiasAddShape — build with BiasAddShape::of(...).
 *   post_bias     PostBiasFn instance (default {}).
 *   bias_offset   tile offset added to all bias reads (default 0). Nonzero when the writer
 *                 pushes the whole per-core bias slice once and the kernel walks it across
 *                 outer iterations; the helper just adds it to the bias tile index.
 *
 * @example  // Row-broadcast bias. The reader pushes Nt bias tiles once; the helper reads
 *           // indices 0..Nt-1 and never touches bias_buf wait/pop (caller-owned).
 *   CircularBuffer partials_buf(cb_partials), bias_buf(cb_bias), out_buf(cb_out);
 *   bias_buf.wait_front(Nt);
 *   add_bias_bcast_rows(partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(Mt, Nt, 1, 1));  // in0_sb, in1_sb, sb_h, sb_w
 *   bias_buf.pop_front(Nt);
 *
 * @example  // TileRowMajor output (must match the upstream matmul_block layout).
 *   add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>(
 *       partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(in0_num_subblocks, in1_num_subblocks,
 *                        out_subblock_h, out_subblock_w, out_block_w));  // last arg = out_row_width
 *
 * @example  // Walk-base: writer pushes the whole per-core bias slice once; the kernel walks
 *           // it via bias_offset advancing per output column-block.
 *   bias_buf.wait_front(bias_ntiles_w);  // caller-managed, fronted across all iters
 *   add_bias_bcast_rows<>(partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w),
 *       {},                  // post_bias (NoPostBias)
 *       bias_block_offset);  // walk-base into the once-pushed bias
 */
template <
    BiasBroadcast broadcast = BiasBroadcast::RowBroadcast,
    OutputCBLayout tile_order = OutputCBLayout::SubblockMajor,
    typename PostBiasFn = bias_add_config::NoPostBias,
    typename Activation = NoneActivation,
    bool in_place = false,
    typename Buf = ::CircularBuffer>
ALWI void add_bias_bcast_rows(
    Buf& partials_buf,
    Buf& bias_buf,
    Buf& out_buf,
    BiasAddShape shape,
    PostBiasFn post_bias = {},
    uint32_t bias_offset = 0);

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"
