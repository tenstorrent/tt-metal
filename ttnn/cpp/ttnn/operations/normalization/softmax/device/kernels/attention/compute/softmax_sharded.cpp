// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

template <uint32_t block_w, uint32_t num_subblocks_w, uint32_t subblock_w>
ALWI void calc_numeric_stable(uint32_t cb_in, uint32_t cb_max_scaler, uint32_t cb_max, uint32_t cb_out) {
    auto cb_in_obj = experimental::CircularBuffer(cb_in);
    auto cb_max_obj = experimental::CircularBuffer(cb_max);
    auto cb_out_obj = experimental::CircularBuffer(cb_out);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            cb_in, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(cb_max);
    cb_max_obj.wait_front(1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    uint32_t index_subblock_w_offset = 0;
    for (uint32_t j = 0; j < num_subblocks_w; j++) {
        tile_regs_acquire();
        cb_out_obj.reserve_back(subblock_w);
        for (uint32_t w = 0; w < subblock_w; w++) {
            uint32_t index = w + index_subblock_w_offset;
            sub_tiles_bcast_cols(cb_in, cb_max, index, 0, w);
        }
        cb_out_obj.reserve_back(subblock_w);
        for (uint32_t w = 0; w < subblock_w; w++) {
            exp_tile<EXP_APPROX>(w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t w = 0; w < subblock_w; w++) {
            pack_tile(w, cb_out);
        }
        tile_regs_release();
        cb_out_obj.push_back(subblock_w);
        index_subblock_w_offset += subblock_w;
    }
    cb_in_obj.pop_front(block_w);
    cb_max_obj.pop_front(1);
    cb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_6);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_max_scaler = tt::CBIndex::c_1;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_13;
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

    auto cb_in0_obj = experimental::CircularBuffer(cb_in0);
    auto cb_max_scaler_obj = experimental::CircularBuffer(cb_max_scaler);
    auto cb_fused_scale_obj = experimental::CircularBuffer(cb_fused_scale);
    auto cb_fused_attn_obj = experimental::CircularBuffer(cb_fused_attn);
    auto cb_exps_obj = experimental::CircularBuffer(cb_exps);
    auto cb_recipsumexps_obj = experimental::CircularBuffer(cb_recipsumexps);
    auto cb_scale_mask_obj = experimental::CircularBuffer(cb_scale_mask);
    auto cb_out0_obj = experimental::CircularBuffer(cb_out0);
    auto cb_x_obj = experimental::CircularBuffer(cb_x);
#ifdef NUMERIC_STABLE
    auto cb_max_obj = experimental::CircularBuffer(cb_max);
#endif

    constexpr int dst0 = 0;
    int index_subblock_w_offset = 0;
    int index = 0;

    for (uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        // fused scale
        reconfig_data_format(cb_in0, cb_fused_scale);
        pack_reconfig_data_format(cb_scale_mask);
        cb_fused_scale_obj.wait_front(1);
        mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            cb_scale_mask_obj.reserve_back(subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, cb_scale_mask);
            }
            tile_regs_release();
            cb_scale_mask_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_in0_obj.pop_front(block_w);
        reconfig_data_format(cb_scale_mask, cb_fused_attn);

        // fused attn
        cb_scale_mask_obj.wait_front(block_w);

#ifndef SHARDED_CAUSAL_MASK
        cb_fused_attn_obj.wait_front(block_w);
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
            tile_regs_acquire();
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
            cb_x_obj.reserve_back(subblock_w);
#ifndef NUMERIC_STABLE
            for (uint32_t w = 0; w < subblock_w; w++) {
                exp_tile<EXP_APPROX>(w);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, cb_x);
            }
            tile_regs_release();
            cb_x_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_scale_mask_obj.pop_front(block_w);

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        cb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w>(cb_x, cb_max_scaler, cb_max, cb_exps);
#endif

#ifdef CAUSAL_MASK
        cb_fused_attn_obj.pop_front(block_w);
#endif
        reconfig_data_format(cb_exps, cb_sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w>(cb_in0, cb_max_scaler, cb_max, cb_exps);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        // exp(x)
        index_subblock_w_offset = 0;
        copy_tile_to_dst_init_short(cb_in0);
        exp_tile_init<EXP_APPROX>();
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                copy_tile(cb_in0, index, w);
            }
            cb_exps_obj.reserve_back(subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                exp_tile<EXP_APPROX>(w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, cb_exps);
            }
            tile_regs_release();
            cb_exps_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_in0_obj.pop_front(block_w);
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        cb_wait_front(cb_exps, block_w);
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(block_w),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t) {
                    recip_tile_init();
                    recip_tile(0);
                });

        // exp(x) / (sum(exp(x)))
        reconfig_data_format(cb_exps, cb_recipsumexps);
        pack_reconfig_data_format(cb_out0);
        cb_recipsumexps_obj.wait_front(1);
        mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            cb_out0_obj.reserve_back(subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recipsumexps, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, cb_out0);
            }
            tile_regs_release();
            cb_out0_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        cb_recipsumexps_obj.pop_front(1);
        cb_exps_obj.pop_front(block_w);
    }
}
