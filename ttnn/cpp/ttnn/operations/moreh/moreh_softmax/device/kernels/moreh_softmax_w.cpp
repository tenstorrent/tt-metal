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

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max via the helper.
            // cb_in0 holds all Wt tiles persistently for later steps, so use
            // WaitUpfrontNoPop — the helper waits for the slice it needs and never pops.
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                cb_in0,
                cb_max_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile (index Wt-1, no pop) and continue reducing
            // into cb_max via Accumulate. The accumulator and output are both cb_max:
            // the helper waits+pops the previous tile, then packs+pushes the new one.
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
        }

        // compute x - max(x)
        dfb_x_m_max_obj.reserve_back(Wt);
        dfb_in0_obj.wait_front(Wt);
        dfb_max_obj.wait_front(1);

        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(dfb_in0_obj, dfb_max_obj);
            sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_x_m_max_obj);
            tile_regs_release();
        }
        dfb_max_obj.pop_front(1);
        dfb_in0_obj.pop_front(Wt);
        dfb_x_m_max_obj.push_back(Wt);

        // compute exp(x - max(x))
        dfb_exps_obj.reserve_back(Wt);
        dfb_x_m_max_obj.wait_front(Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_x_m_max_obj);
            copy_tile(cb_x_m_max, w, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (w == Wt - 1) {
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
        dfb_exps_obj.push_back(Wt);

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
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
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // compute final result
        dfb_out0_obj.reserve_back(Wt);
        dfb_x_m_max_obj.wait_front(Wt);
        dfb_recipsumexps_obj.wait_front(1);

#ifndef LOG
        dfb_exps_obj.wait_front(Wt);
#endif

        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(dfb_x_m_max_obj, dfb_recipsumexps_obj);
            sub_tiles_bcast<BroadcastType::COL>(cb_x_m_max, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_out0_obj);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_cols_init_short_with_dt(dfb_exps_obj, dfb_recipsumexps_obj);
            mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_out0_obj);
            tile_regs_release();
#endif
        }

        dfb_recipsumexps_obj.pop_front(1);
        dfb_x_m_max_obj.pop_front(Wt);
        dfb_out0_obj.push_back(Wt);
#ifndef LOG
        dfb_exps_obj.pop_front(Wt);
#endif
    }
}
