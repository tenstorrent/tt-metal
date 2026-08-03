// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) FPU compute kernel for binary_ng's SCALAR subtile-broadcast binary op.
//
// Port of the CircularBuffer kernels_ng/compute/eltwise_binary_scalar_bcast.cpp, taking the DFB-API + the
// ARCH_QUASAR packer gasket fixes established by the ROW kernel (eltwise_binary_row_bcast_dfb.cpp) and the
// freq/tile_start reuse loop established by the COL kernel (eltwise_binary_col_bcast_dfb.cpp) verbatim.
// SCALAR differs from ROW/COL in what collapses: the broadcasting operand's logical tile is [1,1] (both
// H and W collapse to a single element -- unary_bcast<BroadcastType::SCALAR>, like ROW/COL it lowers to
// the MOVB2D srcB->dest datacopy, differentiated only by SCALAR's broadcast constants: bcast0=1, dst_lo=1,
// num_rows spanning all faces); and the broadcast tile is expanded ONCE per (N,C) slab and REUSED across
// the ENTIRE output for that slab (freq = Ht*Wt) rather than once per tile-row (COL's freq = Wt) or every
// tile (ROW). So the broadcast operand is waited/expanded/popped once per process_tile, its PREPROCESS
// runs once (BCAST_OP), and the OTHER operand streams one tile per freq iteration (OTHER_OP) --
// structurally identical to COL's process_tile, just with a larger freq supplied by the host
// (calculate_compute_kernel_args(SCALAR_*, ...) -> {Ht * Wt, start_t}). The broadcast pass and its two
// pack_reconfig_data_format / ARCH_QUASAR pack_init calls are IDENTICAL to the ROW/COL kernels.
//
// DFB operand naming (SRC_BCAST -> a is the broadcast operand; SRC_BCAST_B -> b is):
//   dfb::pre_lhs (c_0)  dfb::pre_rhs (c_1)  dfb::out (c_2)  dfb::llk_post (c_5 for SRC_BCAST / c_6 for
//   SRC_BCAST_B, the unary_bcast output).  post_lhs/post_rhs (c_3/c_4) exist only when that operand has
//   an activation chain; without activations the post id aliases the pre id, and the broadcast operand's
//   pre id aliases llk_post (the expanded tile feeds the binary op). BCAST_INPUT (0 = LHS bcast,
//   1 = RHS bcast) selects which operand is expanded once (BCAST_OP) vs streamed per iteration (OTHER_OP).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_dfb.hpp"

// One (N,C) slab's worth of work: expand the broadcast operand's single-element tile once into llk_post,
// then run the binary op freq times (reusing the expanded tile) against the streamed OTHER operand. Mirrors
// the CB scalar-bcast process_tile with the DFB API + the ROW/COL kernels' ARCH_QUASAR gasket fixes.
ALWI void process_tile(
    uint32_t dfb_bcast_id,
    uint32_t dfb_llk_post_id,
    uint32_t dfb_pre_lhs_id,
    uint32_t dfb_post_lhs_id,
    uint32_t dfb_pre_rhs_id,
    uint32_t dfb_post_rhs_id,
    uint32_t dfb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // BCAST_OP / OTHER_OP (eltwise_utils_common.hpp) resolve to LHS / RHS off BCAST_INPUT; pair the
    // pre/post DFB ids the same way so the broadcast operand is preprocessed once and the other streams.
#if BCAST_INPUT
    const uint32_t dfb_pre_bcast_id = dfb_pre_rhs_id;
    const uint32_t dfb_post_bcast_id = dfb_post_rhs_id;
    const uint32_t dfb_pre_other_id = dfb_pre_lhs_id;
    const uint32_t dfb_post_other_id = dfb_post_lhs_id;
#else
    const uint32_t dfb_pre_bcast_id = dfb_pre_lhs_id;
    const uint32_t dfb_post_bcast_id = dfb_post_lhs_id;
    const uint32_t dfb_pre_other_id = dfb_pre_rhs_id;
    const uint32_t dfb_post_other_id = dfb_post_rhs_id;
#endif
    DataflowBuffer dfb_bcast(dfb_bcast_id);
    DataflowBuffer dfb_llk_post(dfb_llk_post_id);
    DataflowBuffer dfb_post_bcast(dfb_post_bcast_id);
    DataflowBuffer dfb_post_other(dfb_post_other_id);
    DataflowBuffer dfb_out(dfb_out_id);

    // --- Broadcast pass (ONCE per (N,C) slab): single-element tile -> full tile in llk_post. Identical to
    // the ROW/COL kernels' broadcast pass except for the SCALAR broadcast type. ---
    dfb_bcast.wait_front(num_tiles_per_cycle);
    dfb_llk_post.reserve_back(num_tiles_per_cycle);
    pack_reconfig_data_format(dfb_out_id, dfb_llk_post_id);
#ifdef ARCH_QUASAR
    // On Quasar pack_reconfig_data_format reprograms only the packer format gasket, not the packer
    // destination ring (see reconfig_data_format.h and eltwise_utils_dfb.hpp); retarget the packer to
    // llk_post with pack_init so pack_tile writes there. (unary_bcast_init below also re-inits the packer
    // for llk_post, so this is belt-and-suspenders on the LHS side.)
    pack_init(dfb_llk_post_id);
#endif
    unary_bcast_init<BroadcastType::SCALAR>(dfb_bcast_id, dfb_llk_post_id);

    tile_regs_acquire();
    unary_bcast<BroadcastType::SCALAR>(dfb_bcast_id, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dfb_llk_post_id);
    dfb_llk_post.push_back(num_tiles_per_cycle);
    tile_regs_release();

    pack_reconfig_data_format(dfb_llk_post_id, dfb_out_id);
#ifdef ARCH_QUASAR
    // Retarget the packer destination ring back to dfb_out for the binary-op pack below; without this the
    // gasket-only pack_reconfig above leaves the ring on llk_post and pack_tile(0, out) writes the wrong
    // buffer (the ~constant-output symptom). Mirrors eltwise_utils_dfb.hpp.
    pack_init(dfb_out_id);
#endif
#if defined(ARCH_BLACKHOLE)
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#elif defined(ARCH_QUASAR)
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#endif

    // Broadcast operand's activation chain runs ONCE (its expanded tile is reused across the whole slab).
    PREPROCESS(BCAST_OP, dfb_pre_bcast_id, dfb_post_bcast_id, dfb_out_id, num_tiles_per_cycle);
    dfb_post_bcast.wait_front(num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // OTHER operand streams one tile per iteration.
        PREPROCESS(OTHER_OP, dfb_pre_other_id, dfb_post_other_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_other.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif
        tile_regs_acquire();
        BINARY_OP(dfb_post_lhs_id, dfb_post_rhs_id, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb_out_id);
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_post_other.pop_front(num_tiles_per_cycle);
    }
    dfb_bcast.pop_front(num_tiles_per_cycle);
    dfb_post_bcast.pop_front(num_tiles_per_cycle);
}

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    // The broadcast operand's raw (single-element) tile comes in on its pre_* DFB; unary_bcast expands it
    // into llk_post, and the operand's pre id used by the binary op then aliases llk_post (the expanded
    // tile).
#if SRC_BCAST
    constexpr auto dfb_bcast_id = static_cast<uint32_t>(dfb::pre_lhs);      // c_0: raw a (broadcast operand)
    constexpr auto dfb_llk_post_id = static_cast<uint32_t>(dfb::llk_post);  // c_5: expanded a
    constexpr auto dfb_pre_lhs_id = dfb_llk_post_id;                        // expanded a feeds LHS
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);    // c_1: raw b
#endif
#if SRC_BCAST_B
    constexpr auto dfb_bcast_id = static_cast<uint32_t>(dfb::pre_rhs);      // c_1: raw b (broadcast operand)
    constexpr auto dfb_llk_post_id = static_cast<uint32_t>(dfb::llk_post);  // c_6: expanded b
    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);    // c_0: raw a
    constexpr auto dfb_pre_rhs_id = dfb_llk_post_id;                        // expanded b feeds RHS
#endif

    // post_* DFBs (c_3/c_4) exist only when that operand has an activation chain (guarded like the
    // no-bcast kernel). Without activations the post id aliases the pre id (PREPROCESS is a no-op and the
    // binary op reads the pre DFB directly -- which, for the broadcast operand, is the expanded llk_post).
#if HAS_ACTIVATIONS(LHS)
    constexpr auto dfb_post_lhs_id = static_cast<uint32_t>(dfb::post_lhs);
#else
    constexpr auto dfb_post_lhs_id = dfb_pre_lhs_id;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr auto dfb_post_rhs_id = static_cast<uint32_t>(dfb::post_rhs);
#else
    constexpr auto dfb_post_rhs_id = dfb_pre_rhs_id;
#endif

    binary_op_init_common(dfb_post_lhs_id, dfb_post_rhs_id, dfb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif

    // freq/tile_start reuse loop: freq = Ht * Wt tiles per broadcast (the whole (N,C) slab), tile_start is
    // this core's offset into that slab. The first complete group starts at tile_start; subsequent groups
    // start at 0; a partial trailing group (remaining_iterations) closes the core's tile count. Mirrors the
    // CB scalar-bcast kernel_main and the COL DFB kernel's identical loop shape.
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            dfb_bcast_id,
            dfb_llk_post_id,
            dfb_pre_lhs_id,
            dfb_post_lhs_id,
            dfb_pre_rhs_id,
            dfb_post_rhs_id,
            dfb_out_id,
            tile_freq,
            tile_start,
            num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            dfb_bcast_id,
            dfb_llk_post_id,
            dfb_pre_lhs_id,
            dfb_post_lhs_id,
            dfb_pre_rhs_id,
            dfb_post_rhs_id,
            dfb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle);
    }
}
