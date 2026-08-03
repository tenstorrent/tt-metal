// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) SFPU compute kernel for binary_ng's MIXED (ROW_A_COL_B / ROW_B_COL_A)
// subtile-broadcast binary op.
//
// Copy of eltwise_binary_row_col_bcast_dfb.cpp (the FPU MIXED kernel) with the binary-op section swapped to
// the SFPU path -- the SAME delta eltwise_binary_sfpu_col_bcast_dfb.cpp applies (unary_op_init_common +
// BINARY_SFPU_INIT + the copy-both-operands-to-DST / BINARY_SFPU_OP body, with the ARCH_QUASAR
// copy_tile_to_dst_init_short handling). The HYBRID broadcast is identical to the FPU MIXED kernel: the ROW
// operand is expanded by the compute (unary_bcast<ROW> through llk_post, per output tile); the COL operand
// is expanded by the READER's software-fill (FILL_TILE_WITH_FIRST_COLUMN), delivered pre-expanded once per
// tile-row and reused across the row (freq = Wt). unary_bcast is FPU-side regardless of the binary op that
// consumes its output. This keeps compute at exactly 2 LLK passes (unary_bcast<ROW> + the SFPU binary op) --
// a deliberate reader/compute load-balance; a third unary_bcast<COL> would tip it compute-bound, so the COL
// operand is NOT re-broadcast in compute.
//
// DFB operand naming (BCAST_INPUT: 1 = ROW_A_COL_B [a row, b col], 0 = ROW_B_COL_A [a col, b row]):
//   dfb::pre_lhs (c_0)  dfb::pre_rhs (c_1)  dfb::out (c_2)  dfb::llk_post (c_5 for ROW_A_COL_B / c_6 for
//   ROW_B_COL_A, the unary_bcast<ROW> output for the ROW operand).  post_lhs/post_rhs (c_3/c_4) exist only
//   when that operand has an activation chain. BCAST_INPUT selects which operand is the COL (once-per-row
//   BCAST_OP, reader-filled) vs the ROW (streamed OTHER_OP, unary_bcast<ROW>).

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

// One tile-row's worth of work. The COL operand (BCAST_OP) was delivered pre-expanded by the reader
// software-fill, so it is preprocessed ONCE and reused across the row; the ROW operand (OTHER_OP) streams
// one raw partial tile per iteration and is expanded via unary_bcast<ROW> into llk_post each iteration.
ALWI void process_tile(
    uint32_t dfb_raw_row_id,   // ROW operand's raw reader slot (c_0 for ROW_A_COL_B, c_1 for ROW_B_COL_A)
    uint32_t dfb_llk_post_id,  // expanded ROW operand (c_5 / c_6)
    uint32_t dfb_pre_lhs_id,
    uint32_t dfb_post_lhs_id,
    uint32_t dfb_pre_rhs_id,
    uint32_t dfb_post_rhs_id,
    uint32_t dfb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle ISCLOSE_RT_ARG_PARAMS) {
    using namespace ckernel;

    // BCAST_OP / OTHER_OP (eltwise_utils_common.hpp) resolve to LHS / RHS off BCAST_INPUT. BCAST_OP is the
    // COL operand (reader-filled, preprocessed once); OTHER_OP the ROW operand (streamed, expanded per
    // iteration; its PREPROCESS reads llk_post -- the unary_bcast<ROW> output).
#if BCAST_INPUT                                        // ROW_A_COL_B: COL = b (RHS), ROW = a (LHS)
    const uint32_t dfb_pre_bcast_id = dfb_pre_rhs_id;  // filled COL tile (c_1)
    const uint32_t dfb_post_bcast_id = dfb_post_rhs_id;
    const uint32_t dfb_post_other_id = dfb_post_lhs_id;  // expanded ROW tile's binary-op input (llk_post / c_3)
#else                                                    // ROW_B_COL_A: COL = a (LHS), ROW = b (RHS)
    const uint32_t dfb_pre_bcast_id = dfb_pre_lhs_id;  // filled COL tile (c_0)
    const uint32_t dfb_post_bcast_id = dfb_post_lhs_id;
    const uint32_t dfb_post_other_id = dfb_post_rhs_id;  // expanded ROW tile's binary-op input (llk_post / c_4)
#endif
    DataflowBuffer dfb_raw_row(dfb_raw_row_id);
    DataflowBuffer dfb_llk_post(dfb_llk_post_id);
    DataflowBuffer dfb_post_bcast(dfb_post_bcast_id);
    DataflowBuffer dfb_post_other(dfb_post_other_id);
    DataflowBuffer dfb_out(dfb_out_id);

    // COL operand's activation chain runs ONCE (the reader software-filled the full tile; it is reused
    // across the row). COL is NOT expanded by a compute unary_bcast -- the reader fill is the broadcast
    // (deliberate reader/compute load-balance keeping compute at 2 LLK passes).
    PREPROCESS(BCAST_OP, dfb_pre_bcast_id, dfb_post_bcast_id, dfb_out_id, num_tiles_per_cycle);
    dfb_post_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        // --- ROW broadcast pass (per iteration): the raw partial row tile -> full tile in llk_post.
        // Identical to the single-operand ROW kernel's broadcast pass (unary_bcast<ROW> + the two
        // pack_reconfig_data_format / ARCH_QUASAR pack_init gasket calls); unary_bcast is FPU-side. ---
        dfb_raw_row.wait_front(num_tiles_per_cycle);
        dfb_llk_post.reserve_back(num_tiles_per_cycle);
        pack_reconfig_data_format(dfb_out_id, dfb_llk_post_id);
#ifdef ARCH_QUASAR
        // On Quasar pack_reconfig_data_format reprograms only the packer format gasket, not the packer
        // destination ring (see reconfig_data_format.h and eltwise_utils_dfb.hpp); retarget the packer to
        // llk_post with pack_init so pack_tile writes there.
        pack_init(dfb_llk_post_id);
#endif
        unary_bcast_init<BroadcastType::ROW>(dfb_raw_row_id, dfb_llk_post_id);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb_raw_row_id, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb_llk_post_id);
        dfb_llk_post.push_back(num_tiles_per_cycle);
        tile_regs_release();
        dfb_raw_row.pop_front(num_tiles_per_cycle);

        pack_reconfig_data_format(dfb_llk_post_id, dfb_out_id);
#ifdef ARCH_QUASAR
        // Retarget the packer destination ring back to dfb_out for the binary-op pack below (see above).
        pack_init(dfb_out_id);
#endif
#if defined(ARCH_BLACKHOLE)
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#elif defined(ARCH_QUASAR)
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#endif

        // ROW operand's activation chain (reads the expanded llk_post tile). No-op (post aliases llk_post)
        // when the ROW operand has no activation, in which case the binary op reads llk_post directly.
        PREPROCESS(OTHER_OP, dfb_llk_post_id, dfb_post_other_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_other.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT;
#endif
        tile_regs_acquire();
#ifdef ARCH_QUASAR
        // Quasar's copy_tile_to_dst_init_short_with_dt is a no-op and cannot switch which operand the
        // unpacker reads, so use copy_tile_to_dst_init_short (which reprograms the unpacker descriptor) to
        // point at each operand before its copy_tile loop. matches_metal_v2_slice requires lhs and rhs to
        // share a data format, so the data-format reconfig the WH/BH _with_dt path performs is not needed.
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
    constexpr auto dfb_llk_post_id = static_cast<uint32_t>(dfb::llk_post);  // c_5 (ROW_A_COL_B) / c_6 (ROW_B_COL_A)

    // BCAST_INPUT selects which operand is ROW (compute unary_bcast<ROW>) vs COL (reader software-fill).
    // The ROW operand's raw partial tile arrives on its natural pre DFB slot (c_0 for a, c_1 for b) and is
    // expanded into llk_post, which becomes its binary-op input. The COL operand arrives pre-filled on its
    // pre DFB slot and feeds the binary op directly.
#if BCAST_INPUT                                                           // ROW_A_COL_B: a = ROW (LHS), b = COL (RHS)
    constexpr auto dfb_raw_row_id = static_cast<uint32_t>(dfb::pre_lhs);  // c_0: raw partial a (ROW)
    constexpr auto dfb_pre_lhs_id = dfb_llk_post_id;                      // expanded a feeds LHS
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);  // c_1: reader-filled b (COL)
#else                                                                     // ROW_B_COL_A: a = COL (LHS), b = ROW (RHS)
    constexpr auto dfb_raw_row_id = static_cast<uint32_t>(dfb::pre_rhs);  // c_1: raw partial b (ROW)
    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);  // c_0: reader-filled a (COL)
    constexpr auto dfb_pre_rhs_id = dfb_llk_post_id;                      // expanded b feeds RHS
#endif

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

    // freq/tile_start reuse loop: freq = Wt tiles per COL broadcast, tile_start is the per-core column
    // offset (the same reuse loop the single-operand COL kernel uses).
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            dfb_raw_row_id,
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
            dfb_raw_row_id,
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
