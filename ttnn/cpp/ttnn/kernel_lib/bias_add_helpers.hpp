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
 * add_bias_bcast_rows: row-broadcast bias addition on matmul output.
 *
 * Reads matmul output sub-blocks from partials_cb, adds bias with row broadcast,
 * optionally applies a post-bias operation (e.g., SFPU activation), and packs
 * the result to out_cb.
 *
 * Composes with matmul_block by reading from the same interm_cb that matmul_block
 * packed to (when pack_last_to_interm=true).
 *
 * CB flow: partials_cb (wait+pop per subblock) + bias_cb (wait upfront, no pop)
 *          --> out_cb (reserve+push per subblock)
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release).
 *
 * PREREQUISITE: Caller must handle PACK_RELU configuration, pack format reconfig,
 * and L1_ACC disable before calling this helper.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   partials_cb    CB containing matmul output (= interm_cb from matmul_block).
 *   bias_cb        CB containing bias tiles. One tile per output column, row-broadcast.
 *   out_cb         Output CB for biased result.
 *   PostBiasFn     Functor called per output sub-block after bias addition, before
 *                  packing. (default: NoPostBias)
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_num_subblocks  Number of sub-blocks along M dimension.
 *   in1_num_subblocks  Number of sub-blocks along N dimension.
 *   out_subblock_h     Output sub-block height in tiles.
 *   out_subblock_w     Output sub-block width in tiles.
 *   bias_width_tiles   Number of bias tiles to wait for (= in1_num_subblocks * out_subblock_w).
 *   post_bias          PostBiasFn instance (default: {}).
 *
 * ── Notes ──────────────────────────────────────────────────────────────────
 *
 *   - The helper calls cb_wait_front(bias_cb, bias_width_tiles) internally.
 *   - The helper does NOT pop bias_cb. The caller manages bias tile lifetime.
 *   - The helper performs data format reconfiguration at the start.
 */
template <uint32_t partials_cb, uint32_t bias_cb, uint32_t out_cb, typename PostBiasFn = bias_add_config::NoPostBias>
ALWI void add_bias_bcast_rows(
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t out_subblock_h,
    const uint32_t out_subblock_w,
    const uint32_t bias_width_tiles,
    PostBiasFn post_bias = {});

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"
