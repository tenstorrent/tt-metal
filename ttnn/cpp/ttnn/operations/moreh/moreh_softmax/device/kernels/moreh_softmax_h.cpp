// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    DataflowBuffer dfb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    DataflowBuffer dfb_max_scaler_obj(cb_max_scaler);
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    DataflowBuffer dfb_sum_scaler_obj(cb_sum_scaler);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    DataflowBuffer dfb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    DataflowBuffer dfb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_max = tt::CBIndex::c_26;
    DataflowBuffer dfb_max_obj(cb_max);
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    DataflowBuffer dfb_x_m_max_obj(cb_x_m_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    DataflowBuffer dfb_tmp_obj(cb_tmp);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_out0);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Ht == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                cb_in0,
                cb_max_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        // compute x - max(x)
        dfb_x_m_max_obj.reserve_back(Ht);
        dfb_in0_obj.wait_front(Ht);
        dfb_max_obj.wait_front(1);

        for (uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            sub_bcast_rows_init_short_with_dt(dfb_in0_obj, dfb_max_obj);
            sub_tiles_bcast<BroadcastType::ROW>(cb_in0, cb_max, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_x_m_max_obj);
            tile_regs_release();
        }
        dfb_max_obj.pop_front(1);
        dfb_in0_obj.pop_front(Ht);
        dfb_x_m_max_obj.push_back(Ht);

        // compute exp(x - max(x))
        dfb_exps_obj.reserve_back(Ht);
        dfb_x_m_max_obj.wait_front(Ht);
        for (uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_x_m_max_obj);
            copy_tile(cb_x_m_max, h, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (h == Ht - 1) {
                copy_tile_init_with_dt(dfb_mask_obj);
                copy_tile(cb_mask, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_exps_obj);
            tile_regs_release();
        }
        dfb_exps_obj.push_back(Ht);

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::col(Ht),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::col(Ht),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // compute final result
        dfb_out0_obj.reserve_back(Ht);
        dfb_x_m_max_obj.wait_front(Ht);
        dfb_recipsumexps_obj.wait_front(1);
#ifndef LOG
        dfb_exps_obj.wait_front(Ht);
#endif

        for (uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_rows_init_short_with_dt(dfb_x_m_max_obj, dfb_recipsumexps_obj);
            sub_tiles_bcast<BroadcastType::ROW>(cb_x_m_max, cb_recipsumexps, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_out0_obj);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_rows_init_short_with_dt(dfb_exps_obj, dfb_recipsumexps_obj);
            mul_tiles_bcast_rows(cb_exps, cb_recipsumexps, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_out0_obj);
            tile_regs_release();
#endif
        }

        dfb_recipsumexps_obj.pop_front(1);
        dfb_x_m_max_obj.pop_front(Ht);
        dfb_out0_obj.push_back(Ht);
#ifndef LOG
        dfb_exps_obj.pop_front(Ht);
#endif
    }
}
