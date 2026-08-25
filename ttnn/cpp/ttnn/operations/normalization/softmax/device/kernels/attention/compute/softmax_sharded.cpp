// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "experimental/kernel_args.h"

template <
    std::uint32_t block_w,
    std::uint32_t num_subblocks_w,
    std::uint32_t subblock_w,
    std::uint32_t dfb_in_id,
    std::uint32_t dfb_max_scaler_id,
    std::uint32_t dfb_max_id,
    std::uint32_t dfb_out_id>
ALWI void calc_numeric_stable() {
    DataflowBuffer dfb_in_obj(dfb_in_id);
    DataflowBuffer dfb_max_obj(dfb_max_id);
    DataflowBuffer dfb_out_obj(dfb_out_id);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in_id,
        dfb_max_scaler_id,
        dfb_max_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(dfb_max_id);
    dfb_max_obj.wait_front(1);
    sub_bcast_cols_init(dfb_in_id, dfb_max_id);
    std::uint32_t index_subblock_w_offset = 0;
    for (std::uint32_t j = 0; j < num_subblocks_w; j++) {
        tile_regs_acquire();
        dfb_out_obj.reserve_back(subblock_w);
        for (std::uint32_t w = 0; w < subblock_w; w++) {
            std::uint32_t index = w + index_subblock_w_offset;
            sub_tiles_bcast_cols(dfb_in_id, dfb_max_id, index, 0, w);
        }
        dfb_out_obj.reserve_back(subblock_w);
        for (std::uint32_t w = 0; w < subblock_w; w++) {
            exp_tile<EXP_APPROX>(w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t w = 0; w < subblock_w; w++) {
            pack_tile(w, dfb_out_id);
        }
        tile_regs_release();
        dfb_out_obj.push_back(subblock_w);
        index_subblock_w_offset += subblock_w;
    }
    dfb_in_obj.pop_front(block_w);
    dfb_max_obj.pop_front(1);
    dfb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr std::uint32_t block_h = get_arg(args::block_h);
    constexpr std::uint32_t block_w = get_arg(args::block_w);
    constexpr std::uint32_t subblock_w = get_arg(args::subblock_w);
    constexpr std::uint32_t num_subblocks_w = get_arg(args::num_subblocks_w);

    constexpr auto dfb_in0 = dfb::in0;
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_exps = dfb::exps;
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    constexpr auto dfb_out0 = dfb::out0;

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
#if FUSED_SCALE_MASK
    // fused_scale/fused_attn/scale_mask are bound only on the fused scale-mask path.
    constexpr auto dfb_fused_scale = dfb::fused_scale;
    constexpr auto dfb_fused_attn = dfb::fused_attn;
    constexpr auto dfb_scale_mask = dfb::scale_mask;
#endif
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max = dfb::max;
    constexpr auto dfb_x = dfb::x;
#else
    // Without numeric_stable, dfb_x aliases dfb_exps (Same-FIFO reuse).
    constexpr auto dfb_x = dfb_exps;
#endif

    DataflowBuffer dfb_in0_obj(dfb_in0);
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    DataflowBuffer dfb_exps_obj(dfb_exps);
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    DataflowBuffer dfb_out0_obj(dfb_out0);
    DataflowBuffer dfb_x_obj(dfb_x);
#if FUSED_SCALE_MASK
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    DataflowBuffer dfb_fused_attn_obj(dfb_fused_attn);
    DataflowBuffer dfb_scale_mask_obj(dfb_scale_mask);
#endif

    constexpr int dst0 = 0;
    int index_subblock_w_offset = 0;
    int index = 0;

    for (std::uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        // fused scale
        reconfig_data_format(dfb_in0, dfb_fused_scale);
        pack_reconfig_data_format(dfb_scale_mask);
        dfb_fused_scale_obj.wait_front(1);
        mul_bcast_scalar_init(dfb_in0, dfb_fused_scale);
        index_subblock_w_offset = 0;
        for (std::uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            dfb_scale_mask_obj.reserve_back(subblock_w);
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast_scalar(dfb_in0, dfb_fused_scale, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, dfb_scale_mask);
            }
            tile_regs_release();
            dfb_scale_mask_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        dfb_in0_obj.pop_front(block_w);
        reconfig_data_format(dfb_scale_mask, dfb_fused_attn);

        // fused attn
        dfb_scale_mask_obj.wait_front(block_w);

#ifndef SHARDED_CAUSAL_MASK
        dfb_fused_attn_obj.wait_front(block_w);
#endif

        index_subblock_w_offset = 0;

#ifdef CAUSAL_MASK
        add_init(dfb_scale_mask, dfb_fused_attn);
#else
        add_bcast_rows_init(dfb_scale_mask, dfb_fused_attn);
#endif

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        for (std::uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
#ifdef CAUSAL_MASK
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                add_tiles(dfb_scale_mask, dfb_fused_attn, index, index, w);
            }
#else
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                add_tiles_bcast_rows(dfb_scale_mask, dfb_fused_attn, index, index, w);
            }
#endif
            dfb_x_obj.reserve_back(subblock_w);
#ifndef NUMERIC_STABLE
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                exp_tile<EXP_APPROX>(w);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, dfb_x);
            }
            tile_regs_release();
            dfb_x_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        dfb_scale_mask_obj.pop_front(block_w);

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        dfb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, dfb_x, dfb_max_scaler, dfb_max, dfb_exps>();
#endif

#ifdef CAUSAL_MASK
        dfb_fused_attn_obj.pop_front(block_w);
#endif
        reconfig_data_format(dfb_exps, dfb_sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, dfb_in0, dfb_max_scaler, dfb_max, dfb_exps>();
#else
        reconfig_data_format(dfb_in0, dfb_in0);
        pack_reconfig_data_format(dfb_exps);
        // exp(x)
        index_subblock_w_offset = 0;
        copy_tile_to_dst_init_short(dfb_in0);
        exp_tile_init<EXP_APPROX>();
        for (std::uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                copy_tile(dfb_in0, index, w);
            }
            dfb_exps_obj.reserve_back(subblock_w);
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                exp_tile<EXP_APPROX>(w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, dfb_exps);
            }
            tile_regs_release();
            dfb_exps_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        dfb_in0_obj.pop_front(block_w);
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        dfb_exps_obj.wait_front(block_w);
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(block_w),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        // exp(x) / (sum(exp(x)))
        reconfig_data_format(dfb_exps, dfb_recipsumexps);
        pack_reconfig_data_format(dfb_out0);
        dfb_recipsumexps_obj.wait_front(1);
        mul_bcast_cols_init(dfb_exps, dfb_recipsumexps);
        index_subblock_w_offset = 0;
        for (std::uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            dfb_out0_obj.reserve_back(subblock_w);
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast<BroadcastType::COL>(dfb_exps, dfb_recipsumexps, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t w = 0; w < subblock_w; w++) {
                pack_tile(w, dfb_out0);
            }
            tile_regs_release();
            dfb_out0_obj.push_back(subblock_w);
            index_subblock_w_offset += subblock_w;
        }
        dfb_recipsumexps_obj.pop_front(1);
        dfb_exps_obj.pop_front(block_w);
    }
}
