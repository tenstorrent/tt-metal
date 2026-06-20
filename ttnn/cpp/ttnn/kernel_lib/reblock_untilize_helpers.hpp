// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "internal/mod_div_lib.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // OutputCBLayout

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

/**
 * Data-format reconfig for reblock_and_untilize. Independent compile-time gate from
 * InitUninitMode (mirrors untilize / reduce / matmul_block). reblock reads interm via
 * copy_tile (srcA) and packs to out, so the targets are srcA=interm and pack=out.
 *
 * NoReconfigure              skip both (caller reconfigured externally, or amortized in the
 *                            manual-lifecycle chain).
 * UnpackReconfigure          reconfig_data_format_srca(interm) only.
 * PackReconfigure            pack_reconfig_data_format(out) only.
 * UnpackAndPackReconfigure   (default) both — without it reblock would untilize using
 *                            whatever formats the previous op left.
 */
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure
};

}  // namespace reblock_untilize_config

/**
 * reblock_and_untilize: gather matmul SubblockMajor output into row-major and
 * untilize it in a single pass.
 *
 * Required includes:
 *   #include "api/compute/compute_kernel_hw_startup.h"  // for compute_kernel_hw_startup()
 *   #include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"
 *
 * Consumes the whole output block from `interm_cb` — `in0_num_subblocks` row-groups,
 * each a band of `out_subblock_h` tile rows covering all N-subblocks — and writes
 * row-major untilized output into `out_cb`. Uses pack_untilize_dest per subblock to
 * walk across columns while preserving row ordering. The in0_subblock loop is internal
 * (one call untilizes the whole block, like untilize(num_blocks)); there are no
 * standalone init/uninit wrappers — a single InitAndUninit call covers the lifecycle.
 *
 * INPUT LAYOUT CONSTRAINT: SubblockMajor-only. The interm tile addressing
 * (n * out_subblock_num_tiles + h * out_subblock_w + w) assumes per-subblock grouping.
 * For TileRowMajor input the strip is already tile-row ordered — use the standard untilize
 * helper directly. Enforced via static_assert on the `layout` template parameter.
 *
 * Distinct from the standard untilize helper: untilize reads contiguous tile-rows and so
 * already handles TileRowMajor output directly (do NOT use reblock there). reblock exists
 * for the SubblockMajor case — it gathers subblock-grouped interm tiles into row order
 * while untilizing, which untilize can't express without a strided-input path. It also
 * covers what matmul_block's OutWithUntilize cannot: multi-subblock-wide SubblockMajor
 * output that must be materialized to interm first (e.g. when a phase sits between the
 * matmul and the untilize).
 *
 * Init: by default (InitAndUninit) the helper reconfigs data formats and brackets the
 * whole-block loop with pack_untilize_dest_init + copy_tile_to_dst_init_short / uninit.
 * Reconfig and init/uninit are independent gates (see the two config enums); the caller's
 * only boot responsibility is one compute_kernel_hw_startup(). The partial modes
 * (InitOnly/UninitOnly/Neither) exist only for chaining back-to-back reblock calls.
 *
 * ── Template parameters ──────────────────────────────────────────────────────
 *
 *   out_subblock_w     Subblock width in tiles.
 *   out_block_w        Full output block width in tiles (= out_subblock_w * num_subblocks_w).
 *   init_uninit_mode   InitAndUninit (default), InitOnly, UninitOnly, Neither — see InitUninitMode.
 *   reconfig_mode      UnpackAndPackReconfigure (default) etc. — see ReconfigureRegisterDatatypeMode.
 *   layout             OutputCBLayout of the feeding matmul. Must be SubblockMajor (the only
 *                      supported layout); TileRowMajor fails the static_assert.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_num_subblocks        Number of subblock row-groups along M (the internal loop count).
 *   num_subblocks_w          Number of subblocks along the N dimension.
 *   out_subblock_num_tiles   Tiles per subblock (= out_subblock_h * out_subblock_w).
 *   out_subblock_h           Subblock height in tiles.
 *   interm_buf               Input buffer (tiled, SubblockMajor order — see constraint above).
 *   out_buf                  Output buffer (untilized row-major).
 */
template <
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    reblock_untilize_config::InitUninitMode init_uninit_mode = reblock_untilize_config::InitUninitMode::InitAndUninit,
    reblock_untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    OutputCBLayout layout = OutputCBLayout::SubblockMajor,
    typename Buf = ::CircularBuffer>
inline void reblock_and_untilize(
    uint32_t in0_num_subblocks,
    uint32_t num_subblocks_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    Buf& interm_buf,
    Buf& out_buf);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
