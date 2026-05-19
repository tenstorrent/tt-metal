// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "internal/mod_div_lib.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

namespace reblock_untilize_config {

/**
 * Init/uninit lifecycle for reblock_and_untilize. Mirrors untilize_config::InitUninitMode
 * so callers can use the same idioms across both helpers.
 *
 * InitAndUninit (default) Helper calls pack_untilize_dest_init + copy_tile_to_dst_init_short
 *                         at start and pack_untilize_uninit at end. Use for one-shot calls.
 * InitOnly                Helper calls init at start, skips uninit. Use as the first call in
 *                         a back-to-back chain — pair with Neither (middle) and UninitOnly
 *                         (last) to amortize init/uninit across the chain.
 * UninitOnly              Helper skips init, calls uninit at end. Last call in a chain.
 * Neither                 Helper skips both. Middle calls in a chain — caller is responsible
 *                         for init/uninit at the chain boundaries.
 */
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

}  // namespace reblock_untilize_config

/**
 * Standalone init/uninit wrappers for manual lifecycle control. Mirrors the
 * untilize_init/_uninit pattern so callers running in tight in0_subblock loops
 * can amortize the init across iterations:
 *
 *   reblock_and_untilize_init<...>(interm_buf, out_buf);
 *   for (uint32_t i = 0; i < in0_num_subblocks; ++i) {
 *       reblock_and_untilize<..., InitUninitMode::Neither>(...);
 *   }
 *   reblock_and_untilize_uninit(interm_buf);
 */
template <uint32_t out_subblock_w, uint32_t out_block_w, typename Buf = ::CircularBuffer>
ALWI void reblock_and_untilize_init(Buf& interm_buf, Buf& out_buf);

template <typename Buf = ::CircularBuffer>
ALWI void reblock_and_untilize_uninit(Buf& interm_buf);

/**
 * reblock_and_untilize: gather matmul subblock-order output into row-major and
 * untilize it in a single pass.
 *
 * Required includes:
 *   #include "api/compute/compute_kernel_hw_startup.h"  // for compute_kernel_hw_startup()
 *   #include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"
 *
 * Consumes one "row-group" worth of tiles from `interm_cb` — a band of
 * `out_subblock_h` tile rows covering all N-subblocks in the block — and writes
 * row-major untilized output into `out_cb`. Uses pack_untilize_dest per
 * subblock to walk across columns while preserving row ordering.
 *
 * Init handling: by default the helper calls pack_untilize_dest_init +
 * copy_tile_to_dst_init_short at start and pack_untilize_uninit at end
 * (init_uninit_mode=InitAndUninit). Caller's only init responsibility is one
 * compute_kernel_hw_startup() at boot. For tight in0_subblock loops, switch to
 * InitOnly/Neither/UninitOnly modes — see reblock_untilize_config::InitUninitMode.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   out_subblock_w     Subblock width in tiles.
 *   out_block_w        Full output block width in tiles (= out_subblock_w * num_subblocks_w).
 *   init_uninit_mode   reblock_untilize_config::InitUninitMode: InitAndUninit (default),
 *                      InitOnly, UninitOnly, Neither.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   num_subblocks_w          Number of subblocks along the N dimension.
 *   out_subblock_num_tiles   Tiles per subblock (= out_subblock_h * out_subblock_w).
 *   out_subblock_h           Subblock height in tiles.
 *   interm_buf               Input buffer (tiled, subblock-major order).
 *   out_buf                  Output buffer (untilized row-major).
 */
template <
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    reblock_untilize_config::InitUninitMode init_uninit_mode = reblock_untilize_config::InitUninitMode::InitAndUninit,
    typename Buf = ::CircularBuffer>
inline void reblock_and_untilize(
    uint32_t num_subblocks_w, uint32_t out_subblock_num_tiles, uint32_t out_subblock_h, Buf& interm_buf, Buf& out_buf);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
