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
#include "api/dataflow/circular_buffer.h"

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
    uint32_t cb_in, uint32_t cb_fused_scale_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);
void apply_fused_attn_mask(
    uint32_t cb_in, uint32_t cb_fused_attn_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk, bool do_mask);
void pad_input(uint32_t cb_in, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);
void exp_cb(uint32_t cb_in, uint32_t cb_out, uint32_t cb_max, uint32_t cb_length_t, uint32_t blk);

template <PoolType reduce_type, uint32_t cb_in_id, uint32_t cb_scaler_id, uint32_t cb_prev_out_id, uint32_t cb_out_id>
void reduce_cb(bool use_prev_reduce, uint32_t cb_length_t);
void apply_recip(uint32_t cb_in, uint32_t cb_recip, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);

// CB consumers cannot wrap mid-fifo: pops in one cycle must land exactly on fifo_limit.
// After a partial last pass (Wt % cb_length_t != 0), rd/wr sit at that offset. Push/pop
// `pad` tiles to complete the cycle and return pointers to the CB base before the next stage.
ALWI void realign_cb_after_partial_pass(uint32_t cb_id, uint32_t pad) {
    if (pad == 0) {
        return;
    }
    CircularBuffer cb(cb_id);
    cb.reserve_back(pad);
    cb.push_back(pad);
    cb.wait_front(pad);
    cb.pop_front(pad);
}

ALWI void discard_cb_pad(uint32_t cb_id, uint32_t pad) {
    if (pad == 0) {
        return;
    }
    CircularBuffer cb(cb_id);
    cb.wait_front(pad);
    cb.pop_front(pad);
}

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch
void apply_fused_scale_mask(
    uint32_t cb_in, uint32_t cb_fused_scale_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    // Requirements:
    //   cb_length_t of cb_in and cb_out are the same.
    //   A partial final block (cb_length_t not a multiple of blk) is handled by clamping rem below.
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);
    reconfig_data_format(cb_in, cb_fused_scale_mask);
    pack_reconfig_data_format(cb_out);
    mul_tiles_bcast_scalar_init_short(cb_in, cb_fused_scale_mask);
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        const uint32_t rem = (cur_blk + blk > cb_length_t) ? (cb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        cb_in_obj.wait_front(rem);
        cb_out_obj.reserve_back(rem);
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            mul_tiles_bcast_scalar(cb_in, cb_fused_scale_mask, cur_dst, 0, cur_dst);
        }
        tile_regs_wait();
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_out_obj.push_back(rem);
        cb_in_obj.pop_front(rem);
        tile_regs_release();
    }
}
void apply_fused_attn_mask(
    uint32_t cb_in, uint32_t cb_fused_attn_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk, bool do_mask) {
    auto cb_mask_padded = tt::CBIndex::c_5;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_fused_attn_mask_obj(cb_fused_attn_mask);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_mask_padded_obj(cb_mask_padded);
    reconfig_data_format(cb_in, cb_fused_attn_mask);
    pack_reconfig_data_format(cb_out);
#ifdef CAUSAL_MASK
    add_tiles_init(cb_in, cb_fused_attn_mask);
#else
    add_bcast_rows_init_short(cb_in, cb_fused_attn_mask);
#endif
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        const uint32_t rem = (cur_blk + blk > cb_length_t) ? (cb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        tile_regs_wait();
        cb_in_obj.wait_front(rem);
        cb_fused_attn_mask_obj.wait_front(rem);  // cumulative wait for up to wt tiles
        cb_out_obj.reserve_back(rem);
#ifdef CAUSAL_MASK
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            add_tiles(cb_in, cb_fused_attn_mask, cur_dst, cur_dst, cur_dst);  // tile *= 1/(sum(exp(x)))
        }
#else
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            add_tiles_bcast_rows(cb_in, cb_fused_attn_mask, cur_dst, cur_dst, cur_dst);
        }
#endif
        if (do_mask && cur_blk + rem == cb_length_t) {
            // add mask to the last register to pad with -inf
            reconfig_data_format_srca(cb_mask_padded);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_mask_padded);
            cb_mask_padded_obj.wait_front(1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_mask_padded, 0 /*in_tile_index*/, rem - 1);
        }
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_out_obj.push_back(rem);
        cb_in_obj.pop_front(rem);
        cb_fused_attn_mask_obj.pop_front(rem);
        tile_regs_release();
    }
}

// applies pad to the last pass cb if needed
void pad_input(uint32_t cb_in, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    auto cb_mask_padded = tt::CBIndex::c_5;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_mask_padded_obj(cb_mask_padded);
    reconfig_data_format(cb_in, cb_mask_padded);
    pack_reconfig_data_format(cb_out);
    copy_tile_init(cb_in);  // need to copy from CB to DST to be able to run sfpu math
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        const uint32_t rem = (cur_blk + blk > cb_length_t) ? (cb_length_t - cur_blk) : blk;
        tile_regs_acquire();
        cb_in_obj.wait_front(rem);
        cb_out_obj.reserve_back(rem);
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            if (cur_dst == rem - 1 && cur_blk + rem == cb_length_t) {
                add_tiles_init(cb_in, cb_mask_padded);
                cb_mask_padded_obj.wait_front(1);
                add_tiles(cb_in, cb_mask_padded, cur_dst, 0, cur_dst);
            } else {
                copy_tile(cb_in, cur_dst, cur_dst);
            }
        }
        tile_regs_wait();
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_out_obj.push_back(rem);
        cb_in_obj.pop_front(rem);
        tile_regs_release();
    }
}

void exp_cb(uint32_t cb_in, uint32_t cb_out, uint32_t cb_max, const uint32_t cb_length_t, uint32_t blk) {
    // requirements:
    //   cb_length_t of cb_in and cb_out are the same.
    //   Calculates e^cb_in for cb_length_t num of tiles
    //      Also if numeric stable calcs e^(cb_in- BCASTCOL(cb_max))
    //   A partial final block (cb_length_t not a multiple of blk) is handled by clamping rem below.

    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
#ifdef NUMERIC_STABLE
    reconfig_data_format_srcb(cb_max);
    sub_bcast_cols_init_short(cb_in, cb_max);
#else
    copy_tile_init(cb_in);  // need to copy from CB to DST to be able to run sfpu math
#endif
    exp_tile_init<EXP_APPROX>();
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        const uint32_t rem = (cur_blk + blk > cb_length_t) ? (cb_length_t - cur_blk) : blk;
        cb_in_obj.wait_front(rem);
        cb_out_obj.reserve_back(rem);
        tile_regs_acquire();
#ifdef NUMERIC_STABLE
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            sub_tiles_bcast_cols(cb_in, cb_max, cur_dst, 0, cur_dst);
        }
#else
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            copy_tile(cb_in, cur_dst, cur_dst);
        }
#endif
        cb_in_obj.pop_front(rem);
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            exp_tile<EXP_APPROX>(cur_dst);  // exp on DST[0]
        }
        tile_regs_wait();
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_out_obj.push_back(rem);
        tile_regs_release();
    }
}

template <PoolType reduce_type, uint32_t cb_in_id, uint32_t cb_scaler_id, uint32_t cb_prev_out_id, uint32_t cb_out_id>
void reduce_cb(bool use_prev_reduce, uint32_t cb_length_t) {
    // Single reduce call with lambda that conditionally accumulates
    compute_kernel_lib::reduce<reduce_type, ReduceDim::REDUCE_ROW, cb_in_id, cb_scaler_id, cb_out_id>(
        compute_kernel_lib::ReduceInputBlockShape::row(cb_length_t),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        // PostReduceOp: conditionally accumulate with previous result
        [use_prev_reduce](uint32_t) {
            if (use_prev_reduce) {
                // At this point, DST[0] contains the current reduce result
                // Load previous result into DST[1] and accumulate
                CircularBuffer(cb_prev_out_id).wait_front(1);
                reconfig_data_format_srca(cb_prev_out_id);
                copy_tile_init(cb_prev_out_id);
                copy_tile(cb_prev_out_id, 0, 1);

                // Accumulate based on reduce type
                if constexpr (reduce_type == PoolType::MAX) {
                    binary_max_tile_init();
                    binary_max_tile(0, 1, 0);  // max(DST[0], DST[1]) -> DST[0]
                } else {
                    // SUM reduction
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);  // add(DST[0], DST[1]) -> DST[0]
                }

                CircularBuffer(cb_prev_out_id).pop_front(1);
            }
            // If !use_prev_reduce, lambda is no-op (compiles away)
        });
}

template <PoolType reduce_type, uint32_t cb_in_id, uint32_t cb_scaler_id, uint32_t cb_ping_a, uint32_t cb_ping_b>
ALWI void reduce_cb_pass(uint32_t cur_pass, bool use_prev_reduce, uint32_t cb_length_t) {
    if ((cur_pass & 1) == 0) {
        reduce_cb<reduce_type, cb_in_id, cb_scaler_id, cb_ping_b, cb_ping_a>(use_prev_reduce, cb_length_t);
    } else {
        reduce_cb<reduce_type, cb_in_id, cb_scaler_id, cb_ping_a, cb_ping_b>(use_prev_reduce, cb_length_t);
    }
}

void apply_recip(uint32_t cb_in, uint32_t cb_recip, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_recip_obj(cb_recip);
    CircularBuffer cb_out_obj(cb_out);
    reconfig_data_format(cb_in, cb_recip);
    pack_reconfig_data_format(cb_out);
    cb_recip_obj.wait_front(1);
    mul_bcast_cols_init_short(cb_in, cb_recip);
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        const uint32_t rem = (cur_blk + blk > cb_length_t) ? (cb_length_t - cur_blk) : blk;
        cb_in_obj.wait_front(rem);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            mul_tiles_bcast_cols(cb_in, cb_recip, cur_dst, 0, cur_dst);
        }
        tile_regs_commit();
        cb_out_obj.reserve_back(rem);
        for (uint32_t cur_dst = 0; cur_dst < rem; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_in_obj.pop_front(rem);
        cb_out_obj.push_back(rem);
        tile_regs_release();
    }
}

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    const uint32_t cb_length_t = get_arg_val<uint32_t>(6);

    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_13;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_scale_mask = tt::CBIndex::c_9;
    constexpr auto cb_sumexps = tt::CBIndex::c_7;
    constexpr auto cb_prev_reduce = tt::CBIndex::c_12;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
    constexpr auto cb_max = tt::CBIndex::c_8;
    constexpr auto cb_x = tt::CBIndex::c_10;
    constexpr auto cb_recip = tt::CBIndex::c_16;
    constexpr auto cb_prev_max = tt::CBIndex::c_15;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    CircularBuffer cb_fused_scale_obj(cb_fused_scale);
    CircularBuffer cb_recip_obj(cb_recip);
    CircularBuffer cb_mask_padded_obj(cb_mask_padded);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);
    init_sfpu(cb_mask_padded, cb_mask_padded);

    cb_max_scaler_obj.wait_front(1);  // comes from the reader
    cb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_fused_scale_obj.wait_front(1);
#endif

    uint32_t num_cb_passes = 1 + ((Wt - 1) / cb_length_t);  // ceiling divide
    // Ping-pong reduce outputs: odd num_cb_passes -> cb_max/cb_sumexps, even -> cb_prev_max/cb_prev_reduce
    const uint32_t cb_max_final = (num_cb_passes & 1) ? cb_max : cb_prev_max;
    const uint32_t cb_sum_final = (num_cb_passes & 1) ? cb_sumexps : cb_prev_reduce;
    // Tiles needed after Wt to finish the CB cycle (reader pushes these; we discard/realign).
    const uint32_t cb_align_pad = (cb_length_t - (Wt % cb_length_t)) % cb_length_t;
    // out0 is sized 2*blk; pad so multi-row cores realign (writer drains the same count).
    const uint32_t out0_pad = ((blk * 2) - (Wt % (blk * 2))) % (blk * 2);

    // First loop is to parse and find the sum
    uint32_t dst0 = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // This and all inner loops are for parsing the length of Wt in terms of chunks of the width that can fit in the
        // cb This specific loop is for generaating the sum
        // reads through and finds the max value
        bool use_prev_reduce = false;
        uint32_t length_left_t = Wt;
        uint32_t cur_cb_length_t = cb_length_t;
#ifdef NUMERIC_STABLE
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte max of input values---------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_cb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(cb_in0, cb_fused_scale, cb_scale_mask, cur_cb_length_t, blk);
            apply_fused_attn_mask(cb_scale_mask, cb_fused_attn, cb_x, cur_cb_length_t, blk, do_mask);
            reduce_cb_pass<PoolType::MAX, cb_x, cb_max_scaler, cb_max, cb_prev_max>(
                cur_pass, use_prev_reduce, cur_cb_length_t);
#else
            if (do_mask && cur_pass == num_cb_passes - 1) {
                pad_input(cb_in0, cb_x, cur_cb_length_t, blk);
                reduce_cb_pass<PoolType::MAX, cb_x, cb_max_scaler, cb_max, cb_prev_max>(
                    cur_pass, use_prev_reduce, cur_cb_length_t);
            } else {
                reduce_cb_pass<PoolType::MAX, cb_in0, cb_max_scaler, cb_max, cb_prev_max>(
                    cur_pass, use_prev_reduce, cur_cb_length_t);
            }
#endif
            use_prev_reduce = true;
            length_left_t -= cur_cb_length_t;
            cur_cb_length_t = std::min(cur_cb_length_t, length_left_t);
        }
        // Finish the CB cycle so the next stage starts at fifo base (see realign helpers).
        discard_cb_pad(cb_in0, cb_align_pad);
#if FUSED_SCALE_MASK
        discard_cb_pad(cb_fused_attn, cb_align_pad);
        realign_cb_after_partial_pass(cb_scale_mask, cb_align_pad);
        realign_cb_after_partial_pass(cb_x, cb_align_pad);
#else
        if (mask_padded_data) {
            realign_cb_after_partial_pass(cb_x, cb_align_pad);
        }
#endif
        use_prev_reduce = false;
        length_left_t = Wt;
        cur_cb_length_t = cb_length_t;
#endif
#ifdef NUMERIC_STABLE
        CircularBuffer(cb_max_final).wait_front(1);
#endif

        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte sum of exp values-----------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_cb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(cb_in0, cb_fused_scale, cb_scale_mask, cur_cb_length_t, blk);
            apply_fused_attn_mask(cb_scale_mask, cb_fused_attn, cb_x, cur_cb_length_t, blk, do_mask);
            exp_cb(cb_x, cb_exps, cb_max_final, cur_cb_length_t, blk);
            reduce_cb_pass<PoolType::SUM, cb_exps, cb_sum_scaler, cb_sumexps, cb_prev_reduce>(
                cur_pass, use_prev_reduce, cur_cb_length_t);
#else
            if (do_mask && cur_pass == num_cb_passes - 1) {
                pad_input(cb_in0, cb_x, cur_cb_length_t, blk);
                exp_cb(cb_x, cb_exps, cb_max_final, cur_cb_length_t, blk);
                reduce_cb_pass<PoolType::SUM, cb_exps, cb_sum_scaler, cb_sumexps, cb_prev_reduce>(
                    cur_pass, use_prev_reduce, cur_cb_length_t);
            } else {
                exp_cb(cb_in0, cb_exps, cb_max_final, cur_cb_length_t, blk);
                reduce_cb_pass<PoolType::SUM, cb_exps, cb_sum_scaler, cb_sumexps, cb_prev_reduce>(
                    cur_pass, use_prev_reduce, cur_cb_length_t);
            }
#endif
            use_prev_reduce = true;  // We want to accumulate the previous cb reductions
            length_left_t -= cur_cb_length_t;
            cur_cb_length_t = std::min(cur_cb_length_t, length_left_t);
        }
        discard_cb_pad(cb_in0, cb_align_pad);
        realign_cb_after_partial_pass(cb_exps, cb_align_pad);
#if FUSED_SCALE_MASK
        discard_cb_pad(cb_fused_attn, cb_align_pad);
        realign_cb_after_partial_pass(cb_scale_mask, cb_align_pad);
        realign_cb_after_partial_pass(cb_x, cb_align_pad);
#else
        if (mask_padded_data) {
            realign_cb_after_partial_pass(cb_x, cb_align_pad);
        }
#endif
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------Find denominator 1/∑e^x from ∑e^x-----------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        CircularBuffer(cb_sum_final).wait_front(1);

        reconfig_data_format_srca(cb_sum_final);
        pack_reconfig_data_format(cb_sum_final, cb_recip);
        tile_regs_acquire();
        copy_tile_init(cb_sum_final);
        copy_tile(cb_sum_final, 0, dst0);

        CircularBuffer(cb_sum_final).pop_front(1);

        recip_tile_init();
        recip_tile(dst0);

        tile_regs_commit();
        tile_regs_wait();

        cb_recip_obj.reserve_back(1);
        pack_tile(dst0, cb_recip);
        cb_recip_obj.push_back(1);

        tile_regs_release();

        cb_recip_obj.wait_front(1);
        /*
         * --------------------------------------------------------
         * --------------------------------------------------------
         * ------------------Calcualte final values----------------
         * --------------------------------------------------------
         * --------------------------------------------------------
         */
        length_left_t = Wt;
        cur_cb_length_t = cb_length_t;
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_cb_passes - 1);
#if FUSED_SCALE_MASK
            apply_fused_scale_mask(cb_in0, cb_fused_scale, cb_scale_mask, cur_cb_length_t, blk);
            apply_fused_attn_mask(cb_scale_mask, cb_fused_attn, cb_x, cur_cb_length_t, blk, do_mask);
            exp_cb(cb_x, cb_exps, cb_max_final, cur_cb_length_t, blk);
#else
            if (do_mask && cur_pass == num_cb_passes - 1) {
                pad_input(cb_in0, cb_x, cur_cb_length_t, blk);
                exp_cb(cb_x, cb_exps, cb_max_final, cur_cb_length_t, blk);
            } else {
                exp_cb(cb_in0, cb_exps, cb_max_final, cur_cb_length_t, blk);
            }
#endif
            apply_recip(cb_exps, cb_recip, cb_out0, cur_cb_length_t, blk);
            length_left_t -= cur_cb_length_t;
            cur_cb_length_t = std::min(cur_cb_length_t, length_left_t);
        }
        discard_cb_pad(cb_in0, cb_align_pad);
        realign_cb_after_partial_pass(cb_exps, cb_align_pad);
#if FUSED_SCALE_MASK
        discard_cb_pad(cb_fused_attn, cb_align_pad);
        realign_cb_after_partial_pass(cb_scale_mask, cb_align_pad);
        realign_cb_after_partial_pass(cb_x, cb_align_pad);
#else
        if (mask_padded_data) {
            realign_cb_after_partial_pass(cb_x, cb_align_pad);
        }
#endif
        if (out0_pad > 0) {
            CircularBuffer cb_out0_obj(cb_out0);
            cb_out0_obj.reserve_back(out0_pad);
            cb_out0_obj.push_back(out0_pad);
        }
        cb_recip_obj.pop_front(1);
#ifdef NUMERIC_STABLE
        CircularBuffer(cb_max_final).pop_front(1);
#endif
    }
    cb_mask_padded_obj.pop_front(1);
}  // MAIN
