// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch
void apply_fused_scale_mask(
    uint32_t cb_in, uint32_t cb_fused_scale_mask, uint32_t cb_out, uint32_t cb_length, uint32_t blk) {
    // Requirements:
    //   cb_length of cb_in and cb_out are the same.
    //   blk is a divisor of cb_length
    reconfig_data_format(cb_in, cb_fused_scale_mask);
    pack_reconfig_data_format(cb_out);
    mul_tiles_bcast_scalar_init_short(cb_in, cb_fused_scale_mask);
    for (uint32_t cur_blk = 0; cur_blk < cb_length; cur_blk += blk) {
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
    uint32_t cb_in, uint32_t cb_fused_attn_mask, uint32_t cb_out, uint32_t cb_length, uint32_t blk) {
    reconfig_data_format(cb_in, cb_fused_attn_mask);
    pack_reconfig_data_format(cb_out);
#ifdef CAUSAL_MASK
    add_tiles_init(cb_in, cb_fused_attn_mask);
#else
    add_bcast_rows_init_short(cb_in, cb_fused_attn_mask);
#endif
    for (uint32_t cur_blk = 0; cur_blk < cb_length; cur_blk += blk) {
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

void exp_cb(uint32_t cb_in, uint32_t cb_out, uint32_t cb_length, uint32_t blk) {
    // requirements:
    //   cb_length of cb_in and cb_out are the same.
    //   blk is a divisor of cb_length
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_in, 1);
    UNPACK(tt::compute::common::print_full_tile(cb_in, 0, true));
    for (uint32_t cur_blk = 0; cur_blk < cb_length; cur_blk += blk) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(cb_in, blk);
        cb_reserve_back(cb_out, blk);
        copy_tile_to_dst_init_short(cb_in);
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            copy_tile(cb_in, cur_dst, cur_dst);
        }
        cb_pop_front(cb_in, blk);
        exp_tile_init<EXP_APPROX>();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            exp_tile<EXP_APPROX>(cur_dst);  // exp on DST[0]
        }
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_push_back(cb_out, blk);
        tile_regs_release();
    }
}
void reduce_cb(uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_out, bool use_prev_reduce, uint32_t cb_length) {
    // Requirements:
    //   cb_length of cb_in and cb_out are the same.
    //   blk is a divisor of cb_length
    reconfig_data_format(cb_in, cb_scaler);
    pack_reconfig_data_format(cb_out);
    const uint32_t dst0 = 0;
    const uint32_t dst1 = 1;
    tile_regs_acquire();
    tile_regs_wait();
    reduce_init<true>(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_in, cb_length);
    for (uint32_t cur_tile = 0; cur_tile < cb_length; cur_tile++) {
        reduce_tile(cb_in, cb_scaler, cur_tile, 0, dst0);
    }
    cb_pop_front(cb_in, cb_length);
    if (use_prev_reduce) {
        cb_wait_front(cb_out, 1);
        copy_tile_init(cb_out);
        copy_tile(cb_out, 0, dst1);
        add_binary_tile_init();
        add_binary_tile(dst0, dst1);
        cb_pop_front(cb_out, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    pack_tile(dst0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
void apply_recip(uint32_t cb_in, uint32_t cb_recip, uint32_t cb_out, uint32_t cb_length, uint32_t blk) {
    reconfig_data_format(cb_in, cb_recip);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_recip, 1);
    mul_tiles_bcast_scalar_init_short(cb_in, cb_recip);
    const uint32_t dst0 = 0;
    for (uint32_t cur_blk = 0; cur_blk < cb_length; cur_blk += blk) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(cb_in, blk);
        cb_reserve_back(cb_out, blk);
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            mul_tiles_bcast_scalar(cb_in, cb_recip, cur_blk, 0, dst0);
        }
        tile_regs_commit();
        for (uint32_t cur_dst = 0; cur_dst < blk; cur_dst++) {
            pack_tile(cur_dst, cb_out);
        }
        cb_pop_front(cb_in, blk);
        cb_push_back(cb_out, blk);
        tile_regs_release();
    }
}

namespace NAMESPACE {
void MAIN {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    const uint32_t cb_length = get_arg_val<uint32_t>(6);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_scale_mask = tt::CBIndex::c_9;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_8;
    constexpr auto cb_x = tt::CBIndex::c_10;
#else
    constexpr auto cb_x = cb_exps;
#endif
    uint32_t prev_srca_cb;
    uint32_t prev_srcb_cb;
    uint32_t prev_pack_cb;

    cb_wait_front(cb_scaler, 1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_wait_front(cb_fused_scale, 1);
#endif

    uint32_t num_cb_passes = Wt / cb_length;
    uint32_t residual_cb_length = Wt % cb_length;
    num_cb_passes += residual_cb_length == 0 ? 0 : 1;
    uint32_t print_count = 0;

    // First loop is to parse and find the sum
    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // This and all inner loops are for parsing the length of Wt in terms of chunks of the width that can fit in the
        // cb This specific loop is for generaating the sum
        bool use_prev_reduce = false;
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            // TODO Flesh this out after basic functionality is confirmed.
            // if(cur_pass == num_cb_passes - 1 and residuals_cb_length > 0){
            //      cb_length = residuals_cb_length;
            //      blk = gcd(cb_length, blk);
            // }
            //
            // #if FUSED_SCALE_MASK
            //             apply_fused_scale_mask(cb_fused_scale_in, cb_fused_scale_mask, cb_fused_scale_out, cb_length,
            //             blk);
            // #endif
            // #ifdef CASUAL_MASK
            //             apply_fused_attn_mask(cb_fused_attn_in, cb_fused_attn_mask, cb_attn_out, cb_length, blk);
            // #else
            //          //TODO: Add the non causal variation later
            // #endif
            // #ifdef NUMERIC_STABLE
            //          //TODO: Add function that does a max reduce and then
            // #endif
            // DPRINT << "------------Got here " << print_count << "---------------" << ENDL();
            // print_count++;

            DPRINT << "------------Cb_length: " << cb_length << "---------------" << ENDL();
            DPRINT << "------------blk: " << blk << "---------------" << ENDL();
            reconfig_data_format_srca(cb_in0);
            pack_reconfig_data_format(cb_exps);
            exp_cb(cb_in0, cb_exps, cb_length, blk);
            prev_srca_cb = cb_in0;
            prev_pack_cb = cb_exps;

            reconfig_data_format_srca(prev_srca_cb, cb_exps);
            reconfig_data_format_srcb(cb_scaler);
            pack_reconfig_data_format(prev_pack_cb, cb_recipsumexps);
            reduce_cb(cb_exps, cb_scaler, cb_recipsumexps, use_prev_reduce, cb_length);
            prev_srca_cb = cb_exps;
            prev_srcb_cb = cb_scaler;
            prev_pack_cb = cb_recipsumexps;
            use_prev_reduce = true;  // We want to accumulate the previous cb reductions
        }
        cb_wait_front(cb_recipsumexps, 1);

        reconfig_data_format_srca(prev_srca_cb, cb_recipsumexps);
        // pack_reconfig_data_format(prev_pack_cb, cb_recipsumexps);
        tile_regs_acquire();
        tile_regs_wait();

        copy_tile_init(cb_recipsumexps);
        copy_tile(cb_recipsumexps, 0, dst0);

        cb_pop_front(cb_recipsumexps, 1);

        recip_tile_init();
        recip_tile(dst0);

        tile_regs_commit();

        cb_reserve_back(cb_recipsumexps, 1);
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, 1);

        tile_regs_release();

        // This specific loop is for generating the final values
        cb_wait_front(cb_recipsumexps, 1);
        for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
            reconfig_data_format_srca(prev_srca_cb, cb_in0);
            pack_reconfig_data_format(prev_pack_cb, cb_exps);
            exp_cb(cb_in0, cb_exps, cb_length, blk);
            prev_srca_cb = cb_in0;
            prev_pack_cb = cb_exps;

            reconfig_data_format(prev_srca_cb, cb_in0, prev_srcb_cb, cb_recipsumexps);
            pack_reconfig_data_format(prev_pack_cb, cb_out0);
            apply_recip(prev_pack_cb, cb_recipsumexps, cb_out0, cb_length, blk);
        }
    }
}
}  // namespace NAMESPACE
