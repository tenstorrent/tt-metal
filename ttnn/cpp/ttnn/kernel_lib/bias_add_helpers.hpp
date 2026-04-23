// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

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
 * The `row_broadcast` template param selects the add mode:
 *   - true  (default): row-broadcast — one bias tile per output column, broadcast across
 *                      all M rows of the sub-block. Bias shape is [1, N] / [..., 1, N].
 *   - false          : elementwise — bias has multiple M rows matching the output sub-block.
 *                      Required when bias_padded_shape[-2] == tile_height (see PR #42430).
 *
 * Composes with matmul_block by reading from the same interm_cb that matmul_block
 * packed to (when pack_last_to_interm=true). When the upstream matmul_block runs
 * with row_major_output=true, pass row_major_output=true here too so the bias helper
 * consumes partials_cb and produces out_cb in matching row-major layout; otherwise
 * (default) both layouts are subblock-major.
 *
 * CB flow (row_major_output=false): partials_cb (wait+pop per subblock) + bias_cb
 *          (caller owns wait/pop) --> out_cb (reserve+push per subblock).
 * CB flow (row_major_output=true):  partials_cb (wait+pop per M-row-group) + bias_cb
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
 *   partials_cb       CB containing matmul output (= interm_cb from matmul_block).
 *   bias_cb           CB containing bias tiles. Row-broadcast: one tile per output column.
 *                     Elementwise: multiple M rows per column.
 *   out_cb            Output CB for biased result.
 *   row_broadcast     true for add_tiles_bcast_rows, false for add_tiles. (default: true)
 *   row_major_output  When true, consume partials_cb and produce out_cb in row-major
 *                     layout (matches matmul_block with row_major_output=true). Default
 *                     false uses the legacy subblock-major layout.
 *   PostBiasFn        Functor called per output sub-block after bias addition, before
 *                     packing. (default: NoPostBias)
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_num_subblocks  Number of sub-blocks along M dimension.
 *   in1_num_subblocks  Number of sub-blocks along N dimension.
 *   out_subblock_h     Output sub-block height in tiles.
 *   out_subblock_w     Output sub-block width in tiles.
 *   post_bias          PostBiasFn instance (default: {}).
 *   out_row_width      N-tiles per row of the row-major CB layout. Ignored when
 *                      row_major_output=false. Default 0 derives from
 *                      out_subblock_w * in1_num_subblocks.
 */
template <
    uint32_t partials_cb,
    uint32_t bias_cb,
    uint32_t out_cb,
    bool row_broadcast = true,
    bool row_major_output = false,
    typename PostBiasFn = bias_add_config::NoPostBias>
ALWI void add_bias_bcast_rows(
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    PostBiasFn post_bias = {},
    uint32_t out_row_width = 0);

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"
