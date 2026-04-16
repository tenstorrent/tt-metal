// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "internal/mod_div_lib.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

/**
 * reblock_and_untilize: gather matmul subblock-order output into row-major and
 * untilize it in a single pass.
 *
 * Consumes one "row-group" worth of tiles from `interm_cb` — a band of
 * `out_subblock_h` tile rows covering all N-subblocks in the block — and writes
 * row-major untilized output into `out_cb`. Uses pack_untilize_dest per
 * subblock to walk across columns while preserving row ordering.
 *
 * PREREQUISITE: Caller must call pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb)
 * and copy_tile_to_dst_init_short(interm_cb) once before the first call, and
 * pack_untilize_uninit(interm_cb) after the last call.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   out_subblock_w   Subblock width in tiles.
 *   out_block_w      Full output block width in tiles (= out_subblock_w * num_subblocks_w).
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   num_subblocks_w          Number of subblocks along the N dimension.
 *   out_subblock_num_tiles   Tiles per subblock (= out_subblock_h * out_subblock_w).
 *   out_subblock_h           Subblock height in tiles.
 *   interm_cb_id             Input CB (tiled, subblock-major order).
 *   out_cb_id                Output CB (untilized row-major).
 */
template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_subblocks_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
