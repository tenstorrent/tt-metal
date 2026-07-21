// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) FPU compute kernel for binary_ng's MIXED (ROW_A_COL_B / ROW_B_COL_A)
// subtile-broadcast binary op -- BOTH operands broadcast a DIFFERENT subtile axis at once (one logical
// row, one logical column).
//
// HYBRID, not dual-LLK (a deliberate reader/compute load-balance -- see the reference
// kernels_ng/compute/eltwise_binary_row_col_bcast.cpp): the ROW operand is expanded here by the compute
// (unary_bcast<BroadcastType::ROW> through the intermediate llk_post DFB), while the COL operand is
// expanded by the READER's software-fill (FILL_TILE_WITH_FIRST_COLUMN), delivered pre-expanded on its pre
// DFB. A third unary_bcast<COL> pass in compute would tip the pipeline compute-bound (3 LLK passes); the
// reader-fill keeps compute at exactly 2 LLK passes per output tile (unary_bcast<ROW> + the binary op),
// balancing reader vs compute. The COL operand is therefore NOT re-broadcast in compute -- do not fold it
// into a third unary_bcast. Like the single-operand COL kernel, the COL operand is delivered ONCE per
// tile-row and reused across the row via the freq (= Wt) reuse loop; the ROW operand streams one raw
// partial tile per freq iteration and is re-expanded each iteration.
//
// DFB operand naming (BCAST_INPUT: 1 = ROW_A_COL_B [a row, b col], 0 = ROW_B_COL_A [a col, b row]):
//   dfb::pre_lhs (c_0)  dfb::pre_rhs (c_1)  dfb::out (c_2)  dfb::llk_post (c_5 for ROW_A_COL_B / c_6 for
//   ROW_B_COL_A, the unary_bcast<ROW> output for the ROW operand).  post_lhs/post_rhs (c_3/c_4) exist only
//   when that operand has an activation chain. The COL operand's raw (reader-filled) tile arrives on its
//   pre DFB and feeds the binary op directly (aliased post = pre when no activation); the ROW operand's raw
//   partial tile arrives on the SAME pre DFB slot the reader wrote (c_0 for a, c_1 for b), is expanded into
//   llk_post, and the binary-op input for that operand aliases llk_post. BCAST_INPUT selects which operand
//   is the COL (once-per-row BCAST_OP, reader-filled) vs the ROW (streamed OTHER_OP, unary_bcast<ROW>).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_dfb.hpp"

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
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // BCAST_OP / OTHER_OP (eltwise_utils_common.hpp) resolve to LHS / RHS off BCAST_INPUT. Here BCAST_OP is
    // the COL operand (reader-filled, preprocessed once) and OTHER_OP the ROW operand (streamed, expanded
    // per iteration). The COL operand feeds the binary op straight from its pre/post DFB; the ROW operand
    // feeds it from llk_post (the unary_bcast<ROW> output), so OTHER_OP's PREPROCESS reads llk_post.
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
        // pack_reconfig_data_format / ARCH_QUASAR pack_init gasket calls). ---
        dfb_raw_row.wait_front(num_tiles_per_cycle);
        dfb_llk_post.reserve_back(num_tiles_per_cycle);
        pack_reconfig_data_format(dfb_out_id, dfb_llk_post_id);
#ifdef ARCH_QUASAR
        // On Quasar pack_reconfig_data_format reprograms only the packer format gasket, not the packer
        // destination ring (see reconfig_data_format.h and eltwise_utils_dfb.hpp); retarget the packer to
        // llk_post with pack_init so pack_tile writes there. (unary_bcast_init below also re-inits the
        // packer for llk_post, so this is belt-and-suspenders.)
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
        // Retarget the packer destination ring back to dfb_out for the binary-op pack below; without this
        // the gasket-only pack_reconfig above leaves the ring on llk_post and pack_tile(0, out) writes the
        // wrong buffer (the ~constant-output symptom). Mirrors eltwise_utils_dfb.hpp.
        pack_init(dfb_out_id);
#endif
#if defined(ARCH_BLACKHOLE)
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#elif defined(ARCH_QUASAR)
        PACK((llk_pack_hw_configure(dfb_out_id)));
#endif

        // ROW operand's activation chain (reads the expanded llk_post tile). No-op (post aliases llk_post)
        // when the ROW operand has no activation, in which case the binary op reads llk_post directly.
        PREPROCESS(OTHER_OP, dfb_llk_post_id, dfb_post_other_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_other.wait_front(num_tiles_per_cycle);

        // unary_bcast_init above clobbered the binary-op unpacker config, so re-init the binary op EVERY
        // iteration (mirrors the reference row_col kernel).
        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
        dfb_out.reserve_back(num_tiles_per_cycle);

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

    binary_op_init_common(dfb_post_lhs_id, dfb_post_rhs_id, dfb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

    // freq/tile_start reuse loop: freq = Wt tiles per COL broadcast, tile_start is the per-core column
    // offset (the same reuse loop the single-operand COL kernel uses). The first complete group starts at
    // tile_start; subsequent groups start at 0; a partial trailing group closes the core's tile count.
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
            num_tiles_per_cycle);
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
            num_tiles_per_cycle);
    }
}
