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
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // OutputLayout

namespace compute_kernel_lib {

/**
 * Bias-operand layout in bias_cb selected at compile time.
 *
 * RowBroadcast (default): one bias tile per output column, broadcast across
 *   all M rows of the sub-block via add_tiles_bcast_rows. Logical bias shape
 *   [1, N] / [..., 1, N].
 * Elementwise: bias has multiple M rows matching the output sub-block,
 *   added element-wise via add_tiles. Required when bias_padded_shape[-2]
 *   == tile_height (see main PR #42430).
 */
enum class BiasBroadcast { RowBroadcast, Elementwise };

/**
 * Subblock-grid and row-stride specification for add_bias_bcast_rows.
 *
 * Groups the 4 subblock dims + the row-major row stride into one struct that
 * callers build with BiasAddShape::of(...). Mirrors MatmulBlockShape so the
 * surrounding call site reads consistently when the bias phase follows matmul.
 *
 * out_row_width is only consulted when output_layout == OutputLayout::RowMajor;
 * when 0 (default) the helper derives it from out_subblock_w * in1_num_subblocks.
 */
struct BiasAddShape {
    uint32_t in0_num_subblocks;
    uint32_t in1_num_subblocks;
    uint32_t out_subblock_h;
    uint32_t out_subblock_w;
    uint32_t out_row_width = 0;

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
 * Reads matmul output sub-blocks from partials_cb, adds bias (either row-broadcast
 * or elementwise), optionally applies a post-bias operation (e.g., SFPU activation),
 * and packs the result to out_cb.
 *
 * The `broadcast` template param selects the add mode:
 *   - BiasBroadcast::RowBroadcast (default): one bias tile per output column, broadcast
 *                      across all M rows of the sub-block. Bias shape [1, N] / [..., 1, N].
 *   - BiasBroadcast::Elementwise: bias has multiple M rows matching the output sub-block.
 *                      Required when bias_padded_shape[-2] == tile_height (see PR #42430).
 *
 * Composes with matmul_block by reading from the same interm_cb that matmul_block
 * packed to (when pack_last_to_interm=true). The `output_layout` template must match
 * the upstream matmul_block's layout so the intermediate CB is consumed in the right
 * order.
 *
 * CB flow (layout=SubblockMajor): partials_cb (wait+pop per subblock) + bias_cb
 *          (caller owns wait/pop) --> out_cb (reserve+push per subblock).
 * CB flow (layout=RowMajor):      partials_cb (wait+pop per M-row-group) + bias_cb
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
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   broadcast         BiasBroadcast::RowBroadcast (default, add_tiles_bcast_rows) or
 *                     BiasBroadcast::Elementwise (add_tiles).
 *   output_layout     OutputLayout::SubblockMajor (default, legacy) or OutputLayout::RowMajor.
 *                     Must match the upstream matmul_block's layout.
 *   PostBiasFn        Functor called per output sub-block after bias addition, before
 *                     packing. (default: NoPostBias)
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   partials_buf  Buffer containing matmul output (= interm buffer from matmul_block).
 *   bias_buf      Buffer containing bias tiles. RowBroadcast: one tile per output column.
 *                 Elementwise: multiple M rows per column.
 *   out_buf       Output buffer for biased result.
 *   shape         BiasAddShape — subblock counts, subblock size, row stride.
 *                 Build with BiasAddShape::of(...).
 *   post_bias     PostBiasFn instance (default: {}).
 *   bias_offset   Tile offset added to all bias reads (default: 0). Used when the writer pushes
 *                 the entire per-core bias slice once and the compute kernel walks through it
 *                 across multiple outer iterations (e.g. conv2d's bias_block_offset advancing
 *                 by in1_block_w per output column block). Caller manages the offset; helper
 *                 just adds it to the bias tile index. Default 0 is the bmm pattern (writer
 *                 pushes per outer iter; helper reads from front).
 *
 * @example
 *   // Simple row-broadcast bias, subblock-major output, no activation.
 *   add_bias_bcast_rows(partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(in0_num_subblocks, in1_num_subblocks,
 *                         out_subblock_h, out_subblock_w));
 *
 * @example
 *   // Row-major output (must match upstream matmul_block layout).
 *   // The last BiasAddShape::of arg is out_row_width.
 *   add_bias_bcast_rows<
 *       BiasBroadcast::RowBroadcast,
 *       OutputLayout::RowMajor>(
 *       partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(in0_num_subblocks, in1_num_subblocks,
 *                         out_subblock_h, out_subblock_w,
 *                         out_block_w));
 *
 * @example
 *   // Elementwise bias (multiple M rows) — required when bias is not row-broadcast.
 *   add_bias_bcast_rows<BiasBroadcast::Elementwise>(
 *       partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(in0_num_subblocks, in1_num_subblocks,
 *                         out_subblock_h, out_subblock_w));
 *
 * @example
 *   // Fused SFPU activation after bias via PostBiasFn functor.
 *   add_bias_bcast_rows<
 *       BiasBroadcast::RowBroadcast,
 *       OutputLayout::SubblockMajor,
 *       SFPUPostBias>(partials_buf, bias_buf, out_buf,
 *                      BiasAddShape::of(...), SFPUPostBias{});
 *
 * @example
 *   // conv2d pattern: writer pushes the entire per-core bias slice once at startup;
 *   // compute walks through it via bias_block_offset advancing by in1_block_w per outer
 *   // w-block iteration. Caller waits bias once and never pops; helper reads at offset
 *   // bias_block_offset + per-subblock index.
 *   cb_bias.wait_front(bias_ntiles_w);  // caller-managed, fronted across all outer iters
 *   add_bias_bcast_rows<>(partials_buf, bias_buf, out_buf,
 *                          BiasAddShape::of(in0_num_subblocks, in1_num_subblocks,
 *                                            out_subblock_h, out_subblock_w),
 *                          {},                  // post_bias (NoPostBias)
 *                          bias_block_offset);  // walk-base into the once-pushed bias
 */
template <
    BiasBroadcast broadcast = BiasBroadcast::RowBroadcast,
    OutputLayout output_layout = OutputLayout::SubblockMajor,
    typename PostBiasFn = bias_add_config::NoPostBias,
    typename Buf = ::experimental::CircularBuffer>
ALWI void add_bias_bcast_rows(
    Buf& partials_buf,
    Buf& bias_buf,
    Buf& out_buf,
    BiasAddShape shape,
    PostBiasFn post_bias = {},
    uint32_t bias_offset = 0);

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"
