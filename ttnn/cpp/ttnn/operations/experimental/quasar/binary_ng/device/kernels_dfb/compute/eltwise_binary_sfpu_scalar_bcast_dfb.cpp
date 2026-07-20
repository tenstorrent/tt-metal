// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) SFPU compute kernel for binary_ng's SCALAR subtile-broadcast binary op.
//
// Copy of eltwise_binary_scalar_bcast_dfb.cpp (the FPU SCALAR kernel) with the binary-op section swapped
// to the SFPU path -- the SAME delta eltwise_binary_sfpu_no_bcast_dfb.cpp / eltwise_binary_sfpu_row_bcast_dfb.cpp
// / eltwise_binary_sfpu_col_bcast_dfb.cpp apply (unary_op_init_common + BINARY_SFPU_INIT + the
// copy-both-operands-to-DST / BINARY_SFPU_OP body, with the ARCH_QUASAR copy_tile_to_dst_init_short
// handling). The broadcast pass (unary_bcast<SCALAR> through the intermediate llk_post DFB -- MOVB2D LLK
// datacopy, same as ROW/COL, keyed by SCALAR's broadcast constants dst_lo=1/bcast0=1/num_rows spanning all
// faces) with its two pack_reconfig_data_format / pack_init (ARCH_QUASAR gasket) calls, and the
// freq/tile_start reuse loop (freq = Ht * Wt here, vs COL's Wt), are IDENTICAL to the FPU SCALAR kernel --
// unary_bcast is FPU-side regardless of the binary op that consumes its output.
//
// DFB operand naming (SRC_BCAST -> a is the broadcast operand; SRC_BCAST_B -> b is):
//   dfb::pre_lhs (c_0)  dfb::pre_rhs (c_1)  dfb::out (c_2)  dfb::llk_post (c_5 for SRC_BCAST / c_6 for
//   SRC_BCAST_B, the unary_bcast output).  post_lhs/post_rhs (c_3/c_4) exist only when that operand has
//   an activation chain; without activations the post id aliases the pre id, and the broadcast operand's
//   pre id aliases llk_post (the expanded tile feeds the binary op). BCAST_INPUT (0 = LHS bcast,
//   1 = RHS bcast) selects which operand is expanded once (BCAST_OP) vs streamed per iteration (OTHER_OP).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
// SFPU op headers that ARE ported to Quasar (internally ARCH_QUASAR-aware). bf16 multiply/divide use
// eltwise_binary_sfpu.h; the rest cover the other Quasar-supported SFPU binary ops.
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/binary_comp.h"
// SFPU op headers NOT yet ported to Quasar: they unconditionally pull in WH/BH-only ckernel impls or
// symbols (InstrModLoadStore, DataFormat::UInt32, ckernel_sfpu_div_int32_floor.h, ...) absent from the
// Quasar ckernel tree, so they only compile off-Quasar. The float SFPU binary ops this kernel runs on
// Quasar never use them; exclude them there (mirrors the ARCH_QUASAR guard in eltwise_binary_sfpu.h).
#ifndef ARCH_QUASAR
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/isclose.h"
#endif
#include "api/compute/bcast.h"

#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu_dfb.hpp"

// One (N,C) slab's worth of work: expand the broadcast operand's single-element tile once into llk_post,
// then run the SFPU binary op freq times (reusing the expanded tile) against the streamed OTHER operand.
// Mirrors the CB sfpu scalar-bcast process_tile with the DFB API + the ROW/COL kernels' ARCH_QUASAR
// gasket / SFPU body.
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
    uint32_t num_tiles_per_cycle ISCLOSE_RT_ARG_PARAMS) {
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
    // the FPU SCALAR kernel's broadcast pass. ---
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
    PACK((llk_pack_hw_configure(dfb_out_id)));
#endif

    // Broadcast operand's activation chain runs ONCE (its expanded tile is reused across the whole slab).
    PREPROCESS(BCAST_OP, dfb_pre_bcast_id, dfb_post_bcast_id, dfb_out_id, num_tiles_per_cycle);
    dfb_post_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        // OTHER operand streams one tile per iteration.
        PREPROCESS(OTHER_OP, dfb_pre_other_id, dfb_post_other_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_other.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT;
#endif
        tile_regs_acquire();
#ifdef ARCH_QUASAR
        // Quasar's copy_tile_to_dst_init_short_with_dt is a no-op and cannot switch which operand the
        // unpacker reads, so use copy_tile_to_dst_init_short (which reprograms the unpacker descriptor)
        // to point at each operand before its copy_tile loop. matches_metal_v2_slice requires lhs and rhs
        // to share a data format, so the data-format reconfig the WH/BH _with_dt path performs is not needed.
        copy_tile_to_dst_init_short(dfb_post_lhs_id);
#else
        copy_tile_to_dst_init_short_with_dt(dfb_post_rhs_id, dfb_post_lhs_id);
#endif
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(dfb_post_lhs_id, i, i * 2);
        }
#ifdef ARCH_QUASAR
        copy_tile_to_dst_init_short(dfb_post_rhs_id);
#else
        copy_tile_to_dst_init_short_with_dt(dfb_post_lhs_id, dfb_post_rhs_id);
#endif
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(dfb_post_rhs_id, i, i * 2 + 1);
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT;
#endif
#if ISCLOSE_OP
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2, rtol_bits, atol_bits);
#else
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
#endif
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, dfb_out_id);
        }
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
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg(args::rtol_bits);
    const uint32_t atol_bits = get_arg(args::atol_bits);
#endif

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

    unary_op_init_common(dfb_post_lhs_id, dfb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // freq/tile_start reuse loop: freq = Ht * Wt tiles per broadcast (the whole (N,C) slab), tile_start is
    // this core's offset into that slab. The first complete group starts at tile_start; subsequent groups
    // start at 0; a partial trailing group (remaining_iterations) closes the core's tile count. Mirrors the
    // CB sfpu scalar-bcast kernel_main and the COL DFB kernel's identical loop shape.
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
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
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
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
    }
}
