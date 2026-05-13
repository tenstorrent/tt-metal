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
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

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
    uint32_t in0_num_subblocks;  // Subblock count along M (mirror of MatmulBlockShape::in0_num_subblocks).
    uint32_t in1_num_subblocks;  // Subblock count along N (mirror of MatmulBlockShape::in1_num_subblocks).
    uint32_t out_subblock_h;     // Subblock height in tiles (mirror of MatmulBlockShape::out_subblock_h).
    uint32_t out_subblock_w;     // Subblock width in tiles (mirror of MatmulBlockShape::out_subblock_w).
    uint32_t out_row_width = 0;  // Row stride in tiles for RowMajor pack; ignored when output_layout == SubblockMajor.

    // Argument order mirrors MatmulBlockShape::of's first four args, so the
    // bias call directly under a matmul_block call can reuse the same
    // (in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w)
    // values without renaming. The K-blocking args (in0_block_w, num_k_blocks,
    // batch) on MatmulBlockShape are NOT present here — bias_add has no
    // K dimension.
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
 *                     packing. Math-thread hook (called before tile_regs_commit). Use
 *                     for non-activation math-thread post-bias work; for fused activation
 *                     prefer Activation below — it overlaps with the next math
 *                     iteration and avoids math-thread DST pressure. (default: NoPostBias)
 *   Activation        Default NoneActivation. When the bound activation kind is non-NONE
 *                     the helper fuses SFPU activation onto the PACKER thread (TRISC2)
 *                     at the per-subblock pack stage that follows bias add — replacing
 *                     (not augmenting) the standard tile_regs_wait packer-side sync.
 *                     Mirrors the matmul_block helper's Activation slot; this is where
 *                     activation belongs when the upstream matmul fed into bias
 *                     (FUSE_BIAS path). The ActivationInitHelper init() must be called
 *                     once at kernel boot — either by the upstream matmul_block helper
 *                     (init_mode == Full) or by the caller explicitly when both helpers
 *                     run init_mode == Short. Build an Activation from one of the named
 *                     aliases in sfpu_activation_helpers.hpp (HardtanhActivation<low,
 *                     high>, SeluActivation<alpha, lambda>, …) so the per-activation
 *                     parameter meaning is explicit at the call site; for host-driven
 *                     kernels that read activation + params from compile-time args, wrap
 *                     them as ActivationOp<activation_type, p0, p1, p2>.
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
 *   // Canonical caller-side bias lifecycle. The reader pushes Nt bias tiles
 *   // once per row-broadcast set; the helper reads at indices 0..Nt-1 without
 *   // touching wait/pop on bias_buf. Drive wait/pop on the
 *   // experimental::CircularBuffer object — prefer the buffer object's
 *   // wait_front/pop_front methods over raw cb_wait_front / cb_pop_front so
 *   // the kernel stays uniformly typed against the buffer abstraction.
 *   experimental::CircularBuffer partials_buf(cb_partials);
 *   experimental::CircularBuffer bias_buf(cb_bias);
 *   experimental::CircularBuffer out_buf(cb_out);
 *
 *   bias_buf.wait_front(Nt);
 *   add_bias_bcast_rows<BiasBroadcast::RowBroadcast,
 *                       OutputLayout::SubblockMajor>(
 *       partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(
 *           Mt,    // in0_num_subblocks
 *           Nt,    // in1_num_subblocks
 *           1,     // out_subblock_h
 *           1));   // out_subblock_w
 *   bias_buf.pop_front(Nt);
 *
 * @example
 *   // Minimal row-broadcast bias call shape (assumes buffers and bias_buf
 *   // wait/pop are managed elsewhere — see preceding example for the full
 *   // caller-side lifecycle).
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
    typename Activation = NoneActivation,
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
