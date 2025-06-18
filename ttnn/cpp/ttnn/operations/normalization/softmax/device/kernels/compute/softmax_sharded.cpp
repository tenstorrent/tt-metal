// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

template <uint32_t block_w, uint32_t num_subblocks_w, uint32_t subblock_w>
ALWI void calc_numeric_stable(uint32_t cb_in, uint32_t cb_bcast_scaler, uint32_t cb_max, uint32_t cb_out) {
    // calculate max val per row
    ACQ();
    reconfig_data_format(cb_in, cb_bcast_scaler);
    pack_reconfig_data_format(cb_max);
    cb_reserve_back(cb_max, 1);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, cb_max);
    cb_wait_front(cb_bcast_scaler, 1);
    for (uint32_t w = 0; w < block_w; w++) {
        constexpr uint32_t bcast_scaler0 = 0;
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, w, bcast_scaler0, 0);
    }
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_max);
    pack_tile(0, cb_max);
    cb_push_back(cb_max, 1);
    REL();

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(cb_max);
    cb_wait_front(cb_max, 1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    uint32_t index_subblock_w_offset = 0;
    for (uint32_t j = 0; j < num_subblocks_w; j++) {
        ACQ();
        cb_reserve_back(cb_out, subblock_w);
        for (uint32_t w = 0; w < subblock_w; w++) {
            uint32_t index = w + index_subblock_w_offset;
            sub_tiles_bcast_cols(cb_in, cb_max, index, 0, w);
        }
        cb_reserve_back(cb_out, subblock_w);
        for (uint32_t w = 0; w < subblock_w; w++) {
            exp_tile<EXP_APPROX>(w);
            pack_tile(w, cb_out);
        }
        cb_push_back(cb_out, subblock_w);
        REL();
        index_subblock_w_offset += subblock_w;
    }
    cb_pop_front(cb_in, block_w);
    cb_pop_front(cb_max, 1);
    cb_wait_front(cb_out, block_w);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_6);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_1;
    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    constexpr auto cb_fused_attn = tt::CBIndex::c_3;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_scale_mask = tt::CBIndex::c_8;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_9;
    constexpr auto cb_x = tt::CBIndex::c_10;
#else
    constexpr auto cb_x = cb_exps;
#endif

    constexpr int dst0 = 0;
    int index_subblock_w_offset = 0;
    int index = 0;

    for (uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        // fused scale
        reconfig_data_format(cb_in0, cb_fused_scale);
        pack_reconfig_data_format(cb_scale_mask);
        cb_wait_front(cb_fused_scale, 1);
        mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            cb_reserve_back(cb_scale_mask, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, index, 0, w);
                pack_tile(w, cb_scale_mask);
            }
            cb_push_back(cb_scale_mask, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_in0, block_w);
        reconfig_data_format(cb_scale_mask, cb_fused_attn);

        // fused attn
        cb_wait_front(cb_scale_mask, block_w);

#ifndef SHARDED_CAUSAL_MASK
        cb_wait_front(cb_fused_attn, block_w);
#endif

        index_subblock_w_offset = 0;

#ifdef CAUSAL_MASK
        add_tiles_init(cb_scale_mask, cb_fused_attn);
#else
        add_bcast_rows_init_short(cb_scale_mask, cb_fused_attn);
#endif

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
#ifdef CAUSAL_MASK
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                add_tiles(cb_scale_mask, cb_fused_attn, index, index, w);
            }
#else
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, index, index, w);
            }
#endif
            cb_reserve_back(cb_x, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
#ifndef NUMERIC_STABLE
                exp_tile<EXP_APPROX>(w);
#endif
                pack_tile(w, cb_x);
            }
            cb_push_back(cb_x, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_scale_mask, block_w);

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        cb_wait_front(cb_x, block_w);
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w>(cb_x, cb_bcast_scaler, cb_max, cb_exps);
#endif

#ifdef CAUSAL_MASK
        cb_pop_front(cb_fused_attn, block_w);
#endif
        reconfig_data_format(cb_exps, cb_bcast_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w>(cb_in0, cb_bcast_scaler, cb_max, cb_exps);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        // exp(x)
        index_subblock_w_offset = 0;
        copy_tile_to_dst_init_short(cb_in0);
        exp_tile_init<EXP_APPROX>();
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                copy_tile(cb_in0, index, w);
            }
            cb_reserve_back(cb_exps, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                exp_tile<EXP_APPROX>(w);
                pack_tile(w, cb_exps);
            }
            cb_push_back(cb_exps, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_in0, block_w);
#endif
        reconfig_data_format(cb_exps, cb_bcast_scaler);
#endif  // FUSED_SCALE_MASK

        // sum(exp(x))
        ACQ();
        reduce_init_delta<false>(cb_exps, cb_bcast_scaler, cb_recipsumexps);
        cb_wait_front(cb_exps, block_w);
        cb_wait_front(cb_bcast_scaler, 1);
        cb_reserve_back(cb_recipsumexps, 1);
        for (uint32_t w = 0; w < block_w; w++) {
            constexpr uint32_t bcast_scaler0 = 0;
            reduce_tile(cb_exps, cb_bcast_scaler, w, bcast_scaler0, dst0);
        }
        reduce_revert_delta(cb_recipsumexps);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, 1);
        REL();

        // exp(x) / (sum(exp(x)))
        reconfig_data_format(cb_exps, cb_recipsumexps);
        pack_reconfig_data_format(cb_out0);
        cb_wait_front(cb_recipsumexps, 1);
        mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            cb_reserve_back(cb_out0, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recipsumexps, index, 0, w);
                pack_tile(w, cb_out0);
            }
            cb_push_back(cb_out0, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_exps, block_w);
    }
}
}  // namespace NAMESPACE
