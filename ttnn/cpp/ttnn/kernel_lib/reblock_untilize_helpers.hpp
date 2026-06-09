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
 * Data-format reconfiguration for reblock_and_untilize. Mirrors
 * untilize_config::ReconfigureRegisterDatatypeMode (and the reduce / matmul_block
 * helpers' reconfig switch) so reblock obeys the SAME init paradigm as every other
 * matmul-adjacent helper: the helper guarantees its own init AND its own data-format
 * reconfig; the caller is responsible only for uninit / cleanup of any special
 * unpacker/packer mode a prior op left engaged before calling in.
 *
 * INDEPENDENT of InitUninitMode — fires on its own compile-time gate, exactly like
 * untilize. reblock reads interm via copy_tile (srcA only) and packs untilized output
 * to out, so the two reconfig targets are srcA=interm and pack=out.
 *
 * NoReconfigure              Skip both. Use when the caller reconfigured externally, or
 *                            when the reconfig is amortized via reblock_and_untilize_init
 *                            in the manual-lifecycle (InitOnly/Neither/UninitOnly) pattern
 *                            — the Neither loop calls then pass NoReconfigure.
 * UnpackReconfigure          reconfig_data_format_srca(interm) only.
 * PackReconfigure            pack_reconfig_data_format(out) only.
 * UnpackAndPackReconfigure   (default) Both. The always-correct default: without it
 *                            reblock would untilize using whatever srcA / pack formats the
 *                            previous op (matmul, bias-add) left configured, which is
 *                            correct only when those formats happen to coincide with
 *                            interm / out. Making the reconfig the helper's own
 *                            responsibility removes that latent dependency.
 */
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure
};

}  // namespace reblock_untilize_config

/**
 * Standalone init/uninit wrappers for manual lifecycle control. Mirrors the
 * untilize_init/_uninit pattern so callers running in tight in0_subblock loops
 * can amortize the init AND the data-format reconfig across iterations:
 *
 *   reblock_and_untilize_init<...>(interm_buf, out_buf);   // reconfig + init once
 *   for (uint32_t i = 0; i < in0_num_subblocks; ++i) {
 *       reblock_and_untilize<..., InitUninitMode::Neither,
 *           ReconfigureRegisterDatatypeMode::NoReconfigure>(...);  // neither — _init covered both
 *   }
 *   reblock_and_untilize_uninit(interm_buf);
 *
 * reconfig_mode here selects the same srcA=interm / pack=out reconfig the main
 * reblock_and_untilize does (default UnpackAndPackReconfigure); pass NoReconfigure
 * if the caller already reconfigured externally.
 */
template <
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    reblock_untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    typename Buf = ::CircularBuffer>
ALWI void reblock_and_untilize_init(Buf& interm_buf, Buf& out_buf);

template <typename Buf = ::CircularBuffer>
ALWI void reblock_and_untilize_uninit(Buf& interm_buf);

/**
 * reblock_and_untilize: gather matmul SubblockMajor output into row-major and
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
 * INPUT LAYOUT CONSTRAINT: this helper is SubblockMajor-only. The interm CB
 * tile addressing (n * out_subblock_num_tiles + h * out_subblock_w + w) assumes
 * tiles are grouped per-subblock — subblock(0,0)'s tiles, then subblock(0,1)'s
 * tiles, etc. For TileRowMajor input the row strip is already in tile-row order,
 * so callers should invoke the standard untilize helper directly — no reblock
 * needed. Enforced via static_assert on the `layout` template parameter; the
 * default (SubblockMajor) matches every supported caller and the assert exists
 * to fail compilation cleanly if a future caller pairs TileRowMajor matmul
 * output with this helper by mistake.
 *
 * ── WHY THIS IS A SEPARATE HELPER (not just untilize) ──────────────────────
 * This helper is NOT redundant with the standard `untilize` helper, and it is not
 * a candidate to be folded into it:
 *   • `untilize` reads CONTIGUOUS tile-rows (block_width_tiles wide) and untilizes
 *     them. It has no concept of subblock grouping. So `untilize` already handles
 *     the TileRowMajor matmul-output case directly (see the INPUT LAYOUT CONSTRAINT
 *     above) — for that layout, do NOT use reblock.
 *   • reblock exists for the SubblockMajor case: it GATHERS subblock-grouped interm
 *     tiles (striding block_offset by out_subblock_num_tiles across N-subblocks)
 *     into row order while untilizing. `untilize` cannot express that gather without
 *     a strided-input parameter that would burden its hot, common contiguous path —
 *     so the gather lives here instead.
 *   • It also does NOT overlap with matmul_block's LastBlockTarget::OutWithUntilize,
 *     which untilizes during the matmul pack but is capped to SINGLE-subblock-wide
 *     output (out_block_num_subblocks == 1). reblock's niche is exactly the case
 *     OutWithUntilize cannot cover: MULTI-subblock-wide SubblockMajor output that
 *     must be materialized to interm first (e.g. a fused bias phase sits between the
 *     matmul and the untilize).
 *
 * Init handling: by default the helper reconfigs data formats (srcA=interm,
 * pack=out — see reblock_untilize_config::ReconfigureRegisterDatatypeMode) AND calls
 * pack_untilize_dest_init + copy_tile_to_dst_init_short at start and
 * pack_untilize_uninit at end (init_uninit_mode=InitAndUninit). The reconfig and the
 * init/uninit lifecycle are INDEPENDENT compile-time switches, matching untilize /
 * reduce / matmul_block: the helper owns short init + data-format reconfig; the
 * caller owns uninit of any prior special mode before calling in. Caller's only boot
 * responsibility is one compute_kernel_hw_startup(). For tight in0_subblock loops,
 * amortize via reblock_and_untilize_init (reconfig + init once) and the
 * InitUninitMode::Neither + ReconfigureRegisterDatatypeMode::NoReconfigure loop body.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   out_subblock_w     Subblock width in tiles.
 *   out_block_w        Full output block width in tiles (= out_subblock_w * num_subblocks_w).
 *   init_uninit_mode   reblock_untilize_config::InitUninitMode: InitAndUninit (default),
 *                      InitOnly, UninitOnly, Neither.
 *   reconfig_mode      reblock_untilize_config::ReconfigureRegisterDatatypeMode:
 *                      UnpackAndPackReconfigure (default), UnpackReconfigure,
 *                      PackReconfigure, NoReconfigure. Independent of init_uninit_mode.
 *   layout             OutputCBLayout of the matmul that fed `interm_buf`. Must be
 *                      SubblockMajor (the only layout this helper supports); passing
 *                      TileRowMajor fails the static_assert. Threaded through as a
 *                      template arg so the constraint is explicit at the call site.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
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
    uint32_t num_subblocks_w, uint32_t out_subblock_num_tiles, uint32_t out_subblock_h, Buf& interm_buf, Buf& out_buf);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
