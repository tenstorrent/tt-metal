// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_int_sum.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// clang-format off
// 3 Loops in code
// 1: Optional Max value for numerical stability
//      1: (func: apply_fused_scale_mask) Apply optional fused scale mask followed by (func: apply_fused_attn_mask)
//      apply attention mask
//
//      2: (func: pad_input)Pad tile if step 1 is not done, otherwise -inf padding is done by apply
//      attention mask
// 1: Loop till we have parsed all of WT
// 2: Calculate ∑e^x
//      1: (func: apply_fused_scale_mask) Apply optional fused scale mask followed by (func: apply_fused_attn_mask)
//      apply attention mask
//
//      2: (func: pad_input)Pad tile if step 1 is not done, otherwise -inf padding is done by apply
//      attention mask
//
//      3: (func: exp_cb) calculate cb e^x
//
//      4: (func: reduce_cb) Sums across the width dimension to
//      calculate ∑e^x
// 2: Loop till we have parsed all of WT
// 3: Calculate Final value
//      1: (func: apply_fused_scale_mask) Apply optional fused scale mask followed by apply (func:
//      apply_fused_attn_mask)  attention mask
//
//      2: (func: pad_input) Pad tile if step 1 is not done, otherwise -inf
//      padding is done by apply attention mask
//
//      3: (func: exp_cb) calculate cb e^x
//
//      4: (func: apply_recip) Apply_recip
//      e^x * 1/∑e^x
// 2: Loop till we have parsed all of WT
//clang-format on
void apply_fused_scale_mask(
    std::uint32_t dfb_in, std::uint32_t dfb_fused_scale_mask, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk);
void apply_fused_attn_mask(
    std::uint32_t dfb_in, std::uint32_t dfb_fused_attn_mask, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk, bool do_mask);
void pad_input(std::uint32_t dfb_in, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk);
void exp_cb(std::uint32_t dfb_in, std::uint32_t dfb_out, std::uint32_t dfb_max, std::uint32_t dfb_length_t, std::uint32_t blk);

template <PoolType reduce_type, std::uint32_t dfb_in_id, std::uint32_t dfb_scaler_id, std::uint32_t dfb_prev_out_id, std::uint32_t dfb_out_id>
void reduce_cb(bool use_prev_reduce, std::uint32_t dfb_length_t);
void apply_recip(std::uint32_t dfb_in, std::uint32_t dfb_recip, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk);

// CB consumers cannot wrap mid-fifo: pops in one cycle must land exactly on fifo_limit.
// After a partial last pass (Wt % dfb_length_t != 0), rd/wr sit at that offset. Push/pop
// `pad` tiles to complete the cycle and return pointers to the CB base before the next stage.
// Kept identical to the copy in softmax.cpp.
ALWI void cycle_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(pad);
    dfb.push_back(pad);
    dfb.wait_front(pad);
    dfb.pop_front(pad);
}

// Same, for CBs whose padding tiles the reader already pushed: only consume them.
ALWI void drain_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(dfb_id);
    dfb.wait_front(pad);
    dfb.pop_front(pad);
}

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch
void apply_fused_scale_mask(
    std::uint32_t dfb_in, std::uint32_t dfb_fused_scale_mask, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk) {
    // Requirements:
    //   dfb_length_t of dfb_in and dfb_out are the same.
    //   A partial final block (dfb_length_t not a multiple of blk) is handled by clamping rem below.
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_out_obj(dfb_out);
    reconfig_data_format(dfb_in, dfb_fused_scale_mask);
    pack_reconfig_data_format(dfb_out);
    mul_bcast_scalar_init(dfb_in, dfb_fused_scale_mask);
    for (std::uint32_t cur_blk = 0; cur_blk < dfb_length_t; cur_blk += blk) {
        const std::uint32_t rem = (cur_blk + blk > dfb_length_t) ? (dfb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        dfb_in_obj.wait_front(rem);
        dfb_out_obj.reserve_back(rem);
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            mul_tiles_bcast_scalar(dfb_in, dfb_fused_scale_mask, cur_dst, 0, cur_dst);
        }
        tile_regs_wait();
        tile_regs_commit();
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, dfb_out);
        }
        dfb_out_obj.push_back(rem);
        dfb_in_obj.pop_front(rem);
        tile_regs_release();
    }
}
void apply_fused_attn_mask(
    std::uint32_t dfb_in, std::uint32_t dfb_fused_attn_mask, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk, bool do_mask) {
    constexpr auto dfb_mask_padded = dfb::mask_padded;
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_fused_attn_mask_obj(dfb_fused_attn_mask);
    DataflowBuffer dfb_out_obj(dfb_out);
    DataflowBuffer dfb_mask_padded_obj(dfb_mask_padded);
    reconfig_data_format(dfb_in, dfb_fused_attn_mask);
    pack_reconfig_data_format(dfb_out);
#ifdef CAUSAL_MASK
    add_init(dfb_in, dfb_fused_attn_mask);
#else
    add_bcast_rows_init(dfb_in, dfb_fused_attn_mask);
#endif
    for (std::uint32_t cur_blk = 0; cur_blk < dfb_length_t; cur_blk += blk) {
        const std::uint32_t rem = (cur_blk + blk > dfb_length_t) ? (dfb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        tile_regs_wait();
        dfb_in_obj.wait_front(rem);
        dfb_fused_attn_mask_obj.wait_front(rem);  // cumulative wait for up to wt tiles
        dfb_out_obj.reserve_back(rem);
#ifdef CAUSAL_MASK
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            add_tiles(dfb_in, dfb_fused_attn_mask, cur_dst, cur_dst, cur_dst);  // tile *= 1/(sum(exp(x)))
        }
#else
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            add_tiles_bcast_rows(dfb_in, dfb_fused_attn_mask, cur_dst, cur_dst, cur_dst);
        }
#endif
        if (do_mask && cur_blk + rem == dfb_length_t) {
            // add mask to the last register to pad with -inf
            reconfig_data_format_srca(dfb_mask_padded);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                dfb_mask_padded);
            dfb_mask_padded_obj.wait_front(1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                dfb_mask_padded, 0 /*in_tile_index*/, rem - 1);
        }
        tile_regs_commit();
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, dfb_out);
        }
        dfb_out_obj.push_back(rem);
        dfb_in_obj.pop_front(rem);
        dfb_fused_attn_mask_obj.pop_front(rem);
        tile_regs_release();
    }
}

// applies pad to the last pass cb if needed
void pad_input(std::uint32_t dfb_in, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk) {
    constexpr auto dfb_mask_padded = dfb::mask_padded;
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_out_obj(dfb_out);
    DataflowBuffer dfb_mask_padded_obj(dfb_mask_padded);
    reconfig_data_format(dfb_in, dfb_mask_padded);
    pack_reconfig_data_format(dfb_out);
    copy_init(dfb_in);  // need to copy from CB to DST to be able to run sfpu math
    for (std::uint32_t cur_blk = 0; cur_blk < dfb_length_t; cur_blk += blk) {
        const std::uint32_t rem = (cur_blk + blk > dfb_length_t) ? (dfb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        dfb_in_obj.wait_front(rem);
        dfb_out_obj.reserve_back(rem);
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            if (cur_dst == rem - 1 && cur_blk + rem == dfb_length_t) {
                add_init(dfb_in, dfb_mask_padded);
                dfb_mask_padded_obj.wait_front(1);
                add_tiles(dfb_in, dfb_mask_padded, cur_dst, 0, cur_dst);
            } else {
                copy_tile(dfb_in, cur_dst, cur_dst);
            }
        }
        tile_regs_wait();
        tile_regs_commit();
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, dfb_out);
        }
        dfb_out_obj.push_back(rem);
        dfb_in_obj.pop_front(rem);
        tile_regs_release();
    }
}

void exp_cb(std::uint32_t dfb_in, std::uint32_t dfb_out, std::uint32_t dfb_max, const std::uint32_t dfb_length_t, std::uint32_t blk) {
    // requirements:
    //   dfb_length_t of dfb_in and dfb_out are the same.
    //   Calculates e^dfb_in for dfb_length_t num of tiles
    //      Also if numeric stable calcs e^(dfb_in- BCASTCOL(dfb_max))
    //   A partial final block (dfb_length_t not a multiple of blk) is handled by clamping rem below.

    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_out_obj(dfb_out);
    reconfig_data_format_srca(dfb_in);
    pack_reconfig_data_format(dfb_out);
#ifdef NUMERIC_STABLE
    reconfig_data_format_srcb(dfb_max);
    sub_bcast_cols_init(dfb_in, dfb_max);
#else
    copy_init(dfb_in);  // need to copy from CB to DST to be able to run sfpu math
#endif
    exp_tile_init<EXP_APPROX>();
    for (std::uint32_t cur_blk = 0; cur_blk < dfb_length_t; cur_blk += blk) {
        const std::uint32_t rem = (cur_blk + blk > dfb_length_t) ? (dfb_length_t - cur_blk) : blk;
        dfb_in_obj.wait_front(rem);
        dfb_out_obj.reserve_back(rem);
        tile_regs_acquire();
#ifdef NUMERIC_STABLE
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            sub_tiles_bcast_cols(dfb_in, dfb_max, cur_dst, 0, cur_dst);
        }
#else
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            copy_tile(dfb_in, cur_dst, cur_dst);
        }
#endif
        dfb_in_obj.pop_front(rem);
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            exp_tile<EXP_APPROX>(cur_dst);  // exp on DST[0]
        }
        tile_regs_wait();
        tile_regs_commit();
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, dfb_out);
        }
        dfb_out_obj.push_back(rem);
        tile_regs_release();
    }
}

template <PoolType reduce_type, std::uint32_t dfb_in_id, std::uint32_t dfb_scaler_id, std::uint32_t dfb_prev_out_id, std::uint32_t dfb_out_id>
void reduce_cb(bool use_prev_reduce, std::uint32_t dfb_length_t) {
    // Single reduce call with lambda that conditionally accumulates
    compute_kernel_lib::reduce<reduce_type, ReduceDim::REDUCE_ROW, dfb_in_id, dfb_scaler_id, dfb_out_id>(
        compute_kernel_lib::ReduceInputBlockShape::row(dfb_length_t),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        // PostReduceOp: conditionally accumulate with previous result
        [use_prev_reduce](std::uint32_t) {
            if (use_prev_reduce) {
                // At this point, DST[0] contains the current reduce result
                // Load previous result into DST[1] and accumulate
                DataflowBuffer(dfb_prev_out_id).wait_front(1);
                reconfig_data_format_srca(dfb_prev_out_id);
                copy_init(dfb_prev_out_id);
                copy_tile(dfb_prev_out_id, 0, 1);

                // Accumulate based on reduce type
                if constexpr (reduce_type == PoolType::MAX) {
                    binary_max_tile_init();
                    binary_max_tile(0, 1, 0);  // max(DST[0], DST[1]) -> DST[0]
                } else {
                    // SUM reduction
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);  // add(DST[0], DST[1]) -> DST[0]
                }

                DataflowBuffer(dfb_prev_out_id).pop_front(1);
            }
            // If !use_prev_reduce, lambda is no-op (compiles away)
        });
}

template <PoolType reduce_type, std::uint32_t dfb_in_id, std::uint32_t dfb_scaler_id, std::uint32_t dfb_ping_a, std::uint32_t dfb_ping_b>
ALWI void reduce_dfb_pass(std::uint32_t cur_pass, bool use_prev_reduce, std::uint32_t dfb_length_t) {
    if ((cur_pass & 1) == 0) {
        reduce_cb<reduce_type, dfb_in_id, dfb_scaler_id, dfb_ping_b, dfb_ping_a>(use_prev_reduce, dfb_length_t);
    } else {
        reduce_cb<reduce_type, dfb_in_id, dfb_scaler_id, dfb_ping_a, dfb_ping_b>(use_prev_reduce, dfb_length_t);
    }
}

void apply_recip(std::uint32_t dfb_in, std::uint32_t dfb_recip, std::uint32_t dfb_out, std::uint32_t dfb_length_t, std::uint32_t blk) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_recip_obj(dfb_recip);
    DataflowBuffer dfb_out_obj(dfb_out);
    reconfig_data_format(dfb_in, dfb_recip);
    pack_reconfig_data_format(dfb_out);
    dfb_recip_obj.wait_front(1);
    mul_bcast_cols_init(dfb_in, dfb_recip);
    for (std::uint32_t cur_blk = 0; cur_blk < dfb_length_t; cur_blk += blk) {
        const std::uint32_t rem = (cur_blk + blk > dfb_length_t) ? (dfb_length_t - cur_blk) : blk;
        dfb_in_obj.wait_front(rem);
        tile_regs_acquire();
        tile_regs_wait();
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            mul_tiles_bcast_cols(dfb_in, dfb_recip, cur_dst, 0, cur_dst);
        }
        tile_regs_commit();
        dfb_out_obj.reserve_back(rem);
        for (std::uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, dfb_out);
        }
        dfb_in_obj.pop_front(rem);
        dfb_out_obj.push_back(rem);
        tile_regs_release();
    }
}

void kernel_main() {
    const std::uint32_t NCHt = get_arg(args::num_rows);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t mask_padded_data = get_arg(args::mask_padded_data);
    const std::uint32_t dfb_length_t = get_arg(args::dfb_length);

    // reserve one tile for zeros on dfb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_exps = dfb::exps;
    constexpr auto dfb_sumexps = dfb::recip_sum_exps;
    constexpr auto dfb_prev_reduce = dfb::prev_reduce;
    constexpr auto dfb_in0 = dfb::in0;
    constexpr auto dfb_out0 = dfb::out0;
    constexpr auto dfb_x = dfb::x;
    constexpr auto dfb_recip = dfb::recip;
    constexpr auto dfb_prev_max = dfb::prev_max;
    constexpr auto dfb_mask_padded = dfb::mask_padded;
#if FUSED_SCALE_MASK
    // fused_scale/fused_attn/scale_mask are bound only on the fused scale-mask path.
    constexpr auto dfb_fused_scale = dfb::fused_scale;
    constexpr auto dfb_fused_attn = dfb::fused_attn;
    constexpr auto dfb_scale_mask = dfb::scale_mask;
#endif
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler);
    DataflowBuffer dfb_recip_obj(dfb_recip);
    DataflowBuffer dfb_mask_padded_obj(dfb_mask_padded);
#if FUSED_SCALE_MASK
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
#endif
    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
    copy_init(dfb_mask_padded);

    dfb_max_scaler_obj.wait_front(1);  // comes from the reader
    dfb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.wait_front(1);
#endif

    std::uint32_t num_dfb_passes = 1 + ((Wt - 1) / dfb_length_t);  // ceiling divide
    // Ping-pong reduce outputs: odd num_dfb_passes -> dfb_max/dfb_sumexps, even -> dfb_prev_max/dfb_prev_reduce
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max = dfb::max;
    const std::uint32_t dfb_max_final = (num_dfb_passes & 1) ? (std::uint32_t)dfb_max : (std::uint32_t)dfb_prev_max;
#else
    // dfb_max is only consumed on the numeric-stable path; exp_cb ignores its dfb_max argument otherwise. Bind
    // the unused slot to a valid DFB (dfb_exps) so the handle resolves without dfb_max (c_8) being allocated.
    const std::uint32_t dfb_max_final = dfb_exps;
#endif
    const std::uint32_t dfb_sum_final = (num_dfb_passes & 1) ? (std::uint32_t)dfb_sumexps : (std::uint32_t)dfb_prev_reduce;
    // Tiles needed after Wt to finish each CB's cycle, named for the capacity they align.
    // The streamed CBs all share dfb_length_t; the reader pushes dfb_in0/dfb_fused_attn's pad.
    const std::uint32_t stream_pad = (dfb_length_t - (Wt % dfb_length_t)) % dfb_length_t;
    // out0 is sized 2*blk; pad so multi-row cores realign (writer drains the same count).
    // Uniform blk blocks tile 2*blk exactly, so a Wt that blk divides needs no realignment.
    const std::uint32_t out0_pad = (blk > 0 && (Wt % blk) != 0) ? (((blk * 2) - (Wt % (blk * 2))) % (blk * 2)) : 0;

    // First loop is to parse and find the sum
    std::uint32_t dst0 = 0;

    for (std::uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // This and all inner loops are for parsing the length of Wt in terms of chunks of the width that can fit in the
        // cb This specific loop is for generaating the sum
        // reads through and finds the max value
        bool use_prev_reduce = false;
        std::uint32_t length_left_t = Wt;
        std::uint32_t cur_dfb_length_t = dfb_length_t;
#ifdef NUMERIC_STABLE
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte max of input values---------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        for (std::uint32_t cur_pass = 0; cur_pass < num_dfb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_dfb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(dfb_in0, dfb_fused_scale, dfb_scale_mask, cur_dfb_length_t, blk);
            apply_fused_attn_mask(dfb_scale_mask, dfb_fused_attn, dfb_x, cur_dfb_length_t, blk, do_mask);
            reduce_dfb_pass<PoolType::MAX, dfb_x, dfb_max_scaler, dfb_max, dfb_prev_max>(
                cur_pass, use_prev_reduce, cur_dfb_length_t);
#else
            if (do_mask && cur_pass == num_dfb_passes - 1) {
                pad_input(dfb_in0, dfb_x, cur_dfb_length_t, blk);
                reduce_dfb_pass<PoolType::MAX, dfb_x, dfb_max_scaler, dfb_max, dfb_prev_max>(
                    cur_pass, use_prev_reduce, cur_dfb_length_t);
            } else {
                reduce_dfb_pass<PoolType::MAX, dfb_in0, dfb_max_scaler, dfb_max, dfb_prev_max>(
                    cur_pass, use_prev_reduce, cur_dfb_length_t);
            }
#endif
            use_prev_reduce = true;
            length_left_t -= cur_dfb_length_t;
            cur_dfb_length_t = std::min(cur_dfb_length_t, length_left_t);
        }
        // Finish the CB cycle so the next stage starts at fifo base (see realign helpers).
        drain_dfb_pad(dfb_in0, stream_pad);
#if FUSED_SCALE_MASK
        drain_dfb_pad(dfb_fused_attn, stream_pad);
        cycle_dfb_pad(dfb_scale_mask, stream_pad);
        cycle_dfb_pad(dfb_x, stream_pad);
#else
        if (mask_padded_data) {
            cycle_dfb_pad(dfb_x, stream_pad);
        }
#endif
        use_prev_reduce = false;
        length_left_t = Wt;
        cur_dfb_length_t = dfb_length_t;
#endif
#ifdef NUMERIC_STABLE
        DataflowBuffer(dfb_max_final).wait_front(1);
#endif

        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte sum of exp values-----------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        for (std::uint32_t cur_pass = 0; cur_pass < num_dfb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_dfb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(dfb_in0, dfb_fused_scale, dfb_scale_mask, cur_dfb_length_t, blk);
            apply_fused_attn_mask(dfb_scale_mask, dfb_fused_attn, dfb_x, cur_dfb_length_t, blk, do_mask);
            exp_cb(dfb_x, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
            reduce_dfb_pass<PoolType::SUM, dfb_exps, dfb_sum_scaler, dfb_sumexps, dfb_prev_reduce>(
                cur_pass, use_prev_reduce, cur_dfb_length_t);
#else
            if (do_mask && cur_pass == num_dfb_passes - 1) {
                pad_input(dfb_in0, dfb_x, cur_dfb_length_t, blk);
                exp_cb(dfb_x, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
                reduce_dfb_pass<PoolType::SUM, dfb_exps, dfb_sum_scaler, dfb_sumexps, dfb_prev_reduce>(
                    cur_pass, use_prev_reduce, cur_dfb_length_t);
            } else {
                exp_cb(dfb_in0, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
                reduce_dfb_pass<PoolType::SUM, dfb_exps, dfb_sum_scaler, dfb_sumexps, dfb_prev_reduce>(
                    cur_pass, use_prev_reduce, cur_dfb_length_t);
            }
#endif
            use_prev_reduce = true;  // We want to accumulate the previous cb reductions
            length_left_t -= cur_dfb_length_t;
            cur_dfb_length_t = std::min(cur_dfb_length_t, length_left_t);
        }
        drain_dfb_pad(dfb_in0, stream_pad);
        cycle_dfb_pad(dfb_exps, stream_pad);
#if FUSED_SCALE_MASK
        drain_dfb_pad(dfb_fused_attn, stream_pad);
        cycle_dfb_pad(dfb_scale_mask, stream_pad);
        cycle_dfb_pad(dfb_x, stream_pad);
#else
        if (mask_padded_data) {
            cycle_dfb_pad(dfb_x, stream_pad);
        }
#endif
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------Find denominator 1/∑e^x from ∑e^x-----------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        DataflowBuffer(dfb_sum_final).wait_front(1);

        reconfig_data_format_srca(dfb_sum_final);
        pack_reconfig_data_format(dfb_sum_final, dfb_recip);
        tile_regs_acquire();
        copy_init(dfb_sum_final);
        copy_tile(dfb_sum_final, 0, dst0);

        DataflowBuffer(dfb_sum_final).pop_front(1);

        recip_tile_init();
        recip_tile(dst0);

        tile_regs_commit();
        tile_regs_wait();

        dfb_recip_obj.reserve_back(1);
        pack_tile(dst0, dfb_recip);
        dfb_recip_obj.push_back(1);

        tile_regs_release();

        dfb_recip_obj.wait_front(1);
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte final values----------------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        length_left_t = Wt;
        cur_dfb_length_t = dfb_length_t;
        for (std::uint32_t cur_pass = 0; cur_pass < num_dfb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_dfb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(dfb_in0, dfb_fused_scale, dfb_scale_mask, cur_dfb_length_t, blk);
            apply_fused_attn_mask(dfb_scale_mask, dfb_fused_attn, dfb_x, cur_dfb_length_t, blk, do_mask);
            exp_cb(dfb_x, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
#else
            if (do_mask && cur_pass == num_dfb_passes - 1) {
                pad_input(dfb_in0, dfb_x, cur_dfb_length_t, blk);
                exp_cb(dfb_x, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
            } else {
                exp_cb(dfb_in0, dfb_exps, dfb_max_final, cur_dfb_length_t, blk);
            }
#endif
            apply_recip(dfb_exps, dfb_recip, dfb_out0, cur_dfb_length_t, blk);
            length_left_t -= cur_dfb_length_t;
            cur_dfb_length_t = std::min(cur_dfb_length_t, length_left_t);
        }
        drain_dfb_pad(dfb_in0, stream_pad);
        cycle_dfb_pad(dfb_exps, stream_pad);
#if FUSED_SCALE_MASK
        drain_dfb_pad(dfb_fused_attn, stream_pad);
        cycle_dfb_pad(dfb_scale_mask, stream_pad);
        cycle_dfb_pad(dfb_x, stream_pad);
#else
        if (mask_padded_data) {
            cycle_dfb_pad(dfb_x, stream_pad);
        }
#endif
        if (out0_pad > 0) {
            DataflowBuffer dfb_out0_obj(dfb_out0);
            dfb_out0_obj.reserve_back(out0_pad);
            dfb_out0_obj.push_back(out0_pad);
        }
        dfb_recip_obj.pop_front(1);
#ifdef NUMERIC_STABLE
        DataflowBuffer(dfb_max_final).pop_front(1);
#endif
    }
    dfb_mask_padded_obj.pop_front(1);
}  // MAIN
