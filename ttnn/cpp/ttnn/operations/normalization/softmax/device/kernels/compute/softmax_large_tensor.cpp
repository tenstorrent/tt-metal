// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-Lice
#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_int_sum.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void apply_fused_scale_mask(
    uint32_t cb_in, uint32_t cb_fused_scale_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);
void apply_fused_attn_mask(
    uint32_t cb_in, uint32_t cb_fused_attn_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);
void exp_cb(uint32_t cb_in, uint32_t cb_out, uint32_t cb_max, uint32_t cb_length_t, uint32_t blk, bool do_mask);
template <PoolType reduce_type>
void reduce_cb(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    bool use_prev_reduce,
    uint32_t cb_length_t);
void apply_recip(uint32_t cb_in, uint32_t cb_recip, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk);
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t blk = get_arg_val<uint32_t>(3);
    uint32_t start_ht = get_arg_val<uint32_t>(4);
    uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    uint32_t cb_length_t = get_arg_val<uint32_t>(6);

    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    // auto cb_scaler = tt::CBIndex::c_2;
    auto cb_fused_scale = tt::CBIndex::c_3;
    auto cb_fused_attn = tt::CBIndex::c_4;
    auto cb_exps = tt::CBIndex::c_6;
    auto cb_scale_mask = tt::CBIndex::c_9;
    auto cb_sumexps = tt::CBIndex::c_7;
    auto cb_prev_reduce = tt::CBIndex::c_12;
    auto cb_in0 = tt::CBIndex::c_0;
    auto cb_out0 = tt::CBIndex::c_11;
    auto cb_max = tt::CBIndex::c_8;
    auto cb_recip = tt::CBIndex::c_10;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    constexpr auto cb_mask_padded_bcast = tt::CBIndex::c_13;

    if (mask_padded_data) {
        (DPRINT << "BC" << ENDL());
        tile_regs_acquire();
        reconfig_data_format_srca(cb_mask_padded);
        pack_reconfig_data_format(cb_mask_padded_bcast);
        cb_wait_front(cb_mask_padded, 1);
        unary_bcast_init<BroadcastType::ROW>(cb_mask_padded, cb_mask_padded_bcast);
        unary_bcast<BroadcastType::ROW>(cb_mask_padded, 0, 0);
        tile_regs_wait();
        tile_regs_commit();
        cb_reserve_back(cb_mask_padded_bcast, 1);
        pack_tile(0, cb_mask_padded_bcast);
        cb_push_back(cb_mask_padded_bcast, 1);
        tile_regs_release();
        cb_pop_front(cb_mask_padded, 1);
    }

    cb_wait_front(tt::CBIndex::c_2, 1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_wait_front(cb_fused_scale, 1);
#endif

    uint32_t num_cb_passes = 1 + ((Wt - 1) / cb_length_t);  // ceiling divide
    // (DPRINT << "num_cb_passes: " << num_cb_passes << ENDL());
    // DPRINT << "Wt" << Wt << ENDL();
    // DPRINT << "cb_length_t: " << cb_length_t << ENDL();
    // DPRINT << "blk" <<  blk<< ENDL();

    // First loop is to parse and find the sum
    uint32_t dst0 = 0;
    uint32_t ht = start_ht;

    // binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // This and all inner loops are for parsing the length of Wt in terms of chunks of the width that can fit in the
        // cb This specific loop is for generaating the sum
        bool use_prev_reduce = false;
        uint32_t length_left_t = Wt;
        uint32_t cur_cb_length_t = cb_length_t;
#ifdef NUMERIC_STABLE
        // reads through and finds the max value
        constexpr bool use_prev_reduce_max = false;
        reduce_cb<PoolType::MAX>(cb_in0, tt::CBIndex::c_2, cb_prev_reduce, cb_max, use_prev_reduce_max, Wt);
#endif
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_cb_passes - 1);
            exp_cb(cb_in0, cb_exps, cb_max, cur_cb_length_t, blk, do_mask);

            reduce_cb<PoolType::SUM>(
                cb_exps, tt::CBIndex::c_2, cb_prev_reduce, cb_sumexps, use_prev_reduce, cur_cb_length_t);
            use_prev_reduce = true;  // We want to accumulate the previous cb reductions
            length_left_t -= cur_cb_length_t;
            cur_cb_length_t = std::min(cur_cb_length_t, length_left_t);
            if (cur_pass != num_cb_passes - 1) {
                std::swap(cb_sumexps, cb_prev_reduce);
            }
        }
        cb_wait_front(cb_sumexps, 1);
        // for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
        // }

        // DPRINT << "F0" << ENDL();
        reconfig_data_format_srca(cb_sumexps);
        // DPRINT << "F1" << ENDL();
        pack_reconfig_data_format(cb_sumexps, cb_recip);
        init_sfpu(cb_sumexps, cb_recip);
        tile_regs_acquire();
        copy_tile_init(cb_sumexps);
        copy_tile(cb_sumexps, 0, dst0);

        cb_pop_front(cb_sumexps, 1);

        recip_tile_init();
        recip_tile(dst0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_recip, 1);
        // DPRINT << "F3" << ENDL();
        pack_tile(dst0, cb_recip);
        cb_push_back(cb_recip, 1);

        tile_regs_release();

        cb_wait_front(cb_recip, 1);
        DPRINT << "F4" << ENDL();
        length_left_t = Wt;
        cur_cb_length_t = cb_length_t;
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            bool do_mask = mask_padded_data && (cur_pass == num_cb_passes - 1);
            exp_cb(cb_in0, cb_exps, cb_max, cur_cb_length_t, blk, do_mask);

            apply_recip(cb_exps, cb_recip, cb_out0, cur_cb_length_t, blk);
            length_left_t -= cur_cb_length_t;
            cur_cb_length_t = std::min(cur_cb_length_t, length_left_t);
        }
        cb_pop_front(cb_recip, 1);
        // DPRINT << "DONE COMPUTE ncht: " << ncht << " NCHt: " << NCHt << ENDL();
    }
    // DPRINT << "DONE COMPUTE FINAL " << ENDL();
}

    // for scale+mask+softmax:
    // bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
    // bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
    // Note that the attention mask will not fit in L1 for the entire tensor
    // The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
    // then read another Wt tiles of mask for the next batch
    void
    apply_fused_scale_mask(
        uint32_t cb_in, uint32_t cb_fused_scale_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    // Requirements:
    //   cb_length_t of cb_in and cb_out are the same.
    //   blk is a divisor of cb_length_t
    reconfig_data_format(cb_in, cb_fused_scale_mask);
    pack_reconfig_data_format(cb_out);
    mul_tiles_bcast_scalar_init_short(cb_in, cb_fused_scale_mask);
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(cb_in, blk);
        cb_reserve_back(cb_out, blk);
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            mul_tiles_bcast_scalar(cb_in, cb_fused_scale_mask, cur_dst, 0, cur_dst);
        }
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_fused_scale_mask);
        }
        cb_push_back(cb_out, blk);
        cb_pop_front(cb_in, blk);
        tile_regs_release();
    }
}
void apply_fused_attn_mask(
    uint32_t cb_in, uint32_t cb_fused_attn_mask, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    reconfig_data_format(cb_in, cb_fused_attn_mask);
    pack_reconfig_data_format(cb_out);
#ifdef CAUSAL_MASK
    add_tiles_init(cb_in, cb_fused_attn_mask);
#else
    add_bcast_rows_init_short(cb_in, cb_fused_attn_mask);
#endif
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(cb_in, blk);
        cb_wait_front(cb_fused_attn_mask, blk);  // cumulative wait for up to wt tiles
        cb_reserve_back(cb_out, blk);
#ifdef CAUSAL_MASK
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            add_tiles(cb_in, cb_fused_attn_mask, cur_dst, cur_dst, cur_dst);  // tile *= 1/(sum(exp(x)))
        }
#else
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            // add tile bcast
        }
#endif
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_push_back(cb_out, blk);
        cb_pop_front(cb_in, blk);
        cb_pop_front(cb_fused_attn_mask, blk);
        tile_regs_release();
    }
}

void exp_cb(
    uint32_t cb_in, uint32_t cb_out, uint32_t cb_max, const uint32_t cb_length_t, const uint32_t blk, bool do_mask) {
    // requirements:
    //   cb_length_t of cb_in and cb_out are the same.
    //   blk is a divisor of cb_length_t
    //   Calculates e^cb_in for cb_length_t num of tiles
    //      Also if numeric stable calcs e^(cb_in- BCASTCOL(cb_max))
    constexpr auto cb_mask_padded_bcast = tt::CBIndex::c_13;
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
#ifdef NUMERIC_STABLE
    reconfig_data_format_srcb(cb_max);
    init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(cb_in, cb_max, cb_out);
#else
    copy_tile_init(cb_in);  // need to copy from CB to DST to be able to run sfpu math
#endif
    exp_tile_init<EXP_APPROX>();
    uint32_t loop = 0;
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        cb_wait_front(cb_in, blk);
        cb_reserve_back(cb_out, blk);
        tile_regs_acquire();
#ifdef NUMERIC_STABLE
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            sub_tiles_bcast_cols(cb_in, cb_max, cur_dst, 0, cur_dst);
        }
#else
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            copy_tile(cb_in, cur_dst, cur_dst);
        }
#endif
        if (do_mask && cur_blk == cb_length_t - blk) {
            reconfig_data_format_srca(cb_mask_padded_bcast);
            binary_dest_reuse_tiles_init(cb_mask_padded_bcast);
            cb_wait_front(cb_mask_padded_bcast, 1);
            binary_dest_reuse_tiles(cb_mask_padded_bcast, 0, blk - 1);
            exp_tile_init<EXP_APPROX>();
        }
        cb_pop_front(cb_in, blk);
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            exp_tile<EXP_APPROX>(cur_dst);  // exp on DST[0]
        }
        tile_regs_wait();
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_push_back(cb_out, blk);
        tile_regs_release();
    }
}

template <PoolType reduce_type>
void reduce_cb(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    bool use_prev_reduce,
    uint32_t cb_length_t) {
    // Requirements:
    //   blk is a divisor of cb_length_t reconfig_data_format(cb_in, cb_scaler);
    //   len(Data) fed into cb_in, does not need all at once== cb_length_t
    //   len(cb_out) == 1

    reconfig_data_format(cb_in, cb_scaler);
    pack_reconfig_data_format(cb_out);
    tile_regs_acquire();
    cb_reserve_back(cb_out, 1);
    cb_wait_front(cb_scaler, 1);
    reduce_init_delta<false, reduce_type, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out);
    for (uint32_t cur_tile = 0; cur_tile < cb_length_t; cur_tile++) {
        cb_wait_front(cb_in, 1);
        reduce_tile<reduce_type, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
        cb_pop_front(cb_in, 1);
    }
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_out);

    if (use_prev_reduce) {
        reconfig_data_format_srca(cb_prev_out);
        init_sfpu(cb_prev_out, cb_out);
        cb_wait_front(cb_prev_out, 1);
        copy_tile_init(cb_prev_out);
        copy_tile(cb_prev_out, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1);
        cb_pop_front(cb_prev_out, 1);
    }
    dprint_tensix_dest_reg(0);
    tile_regs_wait();
    tile_regs_commit();
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    cb_wait_front(cb_out, 1);
    UNPACK(tt::compute::common::print_full_tile(cb_out, 0, true));
    tile_regs_release();
}
void apply_recip(uint32_t cb_in, uint32_t cb_recip, uint32_t cb_out, uint32_t cb_length_t, uint32_t blk) {
    reconfig_data_format(cb_in, cb_recip);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_recip, 1);
    mul_bcast_cols_init_short(cb_in, cb_recip);
    for (uint32_t cur_blk = 0; cur_blk < cb_length_t; cur_blk += blk) {
        cb_wait_front(cb_in, blk);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            mul_tiles_bcast_cols(cb_in, cb_recip, cur_dst, 0, cur_dst);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, blk);
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_pop_front(cb_in, blk);
        cb_push_back(cb_out, blk);
        tile_regs_release();
    }
}

}  // namespace NAMESPACE
