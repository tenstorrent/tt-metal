// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    experimental::CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    experimental::CircularBuffer cb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    experimental::CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    experimental::CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    experimental::CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    experimental::CircularBuffer cb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    experimental::CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_max = tt::CBIndex::c_26;
    experimental::CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    experimental::CircularBuffer cb_x_m_max_obj(cb_x_m_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    experimental::CircularBuffer cb_tmp_obj(cb_tmp);

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: bulk reduce of Wt-1 full tiles into cb_max (pack preserves full DEST).
            cb_max_obj.reserve_back(1);

            tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_in0, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, cb_max);
            for (uint32_t x = 0; x < Wt - 1; ++x) {
                cb_in0_obj.wait_front(x + 1);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, x, 0, dst0);
            }
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_max_obj.push_back(1);

            // Phase 2: merge the masked last tile into cb_max.
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

            cb_max_obj.wait_front(1);
            cb_tmp_obj.wait_front(1);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_max);
            copy_tile(cb_max, 0, dst0);

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_tmp, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_max_scaler, cb_max);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_max_scaler, 0, 0, dst0);
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_max_obj.pop_front(1);
            cb_tmp_obj.pop_front(1);
            cb_max_obj.push_back(1);
        }

        // compute x - max(x)
        cb_x_m_max_obj.reserve_back(Wt);
        cb_in0_obj.wait_front(Wt);
        cb_max_obj.wait_front(1);

        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_x_m_max);
            tile_regs_release();
        }
        cb_max_obj.pop_front(1);
        cb_in0_obj.pop_front(Wt);
        cb_x_m_max_obj.push_back(Wt);

        // compute exp(x - max(x))
        cb_exps_obj.reserve_back(Wt);
        cb_x_m_max_obj.wait_front(Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            copy_tile_init_with_dt(cb_x_m_max);
            copy_tile(cb_x_m_max, w, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (w == Wt - 1) {
                copy_tile_init_with_dt(cb_mask);
                copy_tile(cb_mask, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_exps);
            tile_regs_release();
        }
        cb_exps_obj.push_back(Wt);

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(Wt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(Wt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // compute final result
        cb_out0_obj.reserve_back(Wt);
        cb_x_m_max_obj.wait_front(Wt);
        cb_recipsumexps_obj.wait_front(1);

#ifndef LOG
        cb_exps_obj.wait_front(Wt);
#endif

        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_x_m_max, cb_recipsumexps);
            sub_tiles_bcast<BroadcastType::COL>(cb_x_m_max, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_cols_init_short_with_dt(cb_exps, cb_recipsumexps);
            mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#endif
        }

        cb_recipsumexps_obj.pop_front(1);
        cb_x_m_max_obj.pop_front(Wt);
        cb_out0_obj.push_back(Wt);
#ifndef LOG
        cb_exps_obj.pop_front(Wt);
#endif
    }
}
