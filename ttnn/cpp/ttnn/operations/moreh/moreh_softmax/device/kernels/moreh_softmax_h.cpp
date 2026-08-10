// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto dfb_in0 = dfb::in0;
    DataflowBuffer dfb_in0_obj(dfb_in0);
    constexpr auto dfb_mask = dfb::mask;
    DataflowBuffer dfb_mask_obj(dfb_mask);
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler);
    constexpr auto dfb_out0 = dfb::out0;
    DataflowBuffer dfb_out0_obj(dfb_out0);
    constexpr auto dfb_exps = dfb::exps;
    DataflowBuffer dfb_exps_obj(dfb_exps);
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    constexpr auto dfb_max = dfb::max;
    DataflowBuffer dfb_max_obj(dfb_max);
    constexpr auto dfb_x_m_max = dfb::x_minus_max;
    DataflowBuffer dfb_x_m_max_obj(dfb_x_m_max);
    constexpr auto dfb_tmp = dfb::tmp;
    DataflowBuffer dfb_tmp_obj(dfb_tmp);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr std::uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_out0);

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Ht loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    std::uint32_t N = get_arg(args::N);
    std::uint32_t Ht = get_arg(args::Ht);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Ht == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp, dfb_max_scaler, dfb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                dfb_in0,
                dfb_max_scaler,
                dfb_max,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp, dfb_max_scaler, dfb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb_max, 1));  // iteration=1, reload from dfb_max
        }

        // compute x - max(x)
        dfb_x_m_max_obj.reserve_back(Ht);
        dfb_in0_obj.wait_front(Ht);
        dfb_max_obj.wait_front(1);

        for (std::uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            sub_bcast_rows_init_with_dt(dfb_in0_obj, dfb_max_obj);
            sub_tiles_bcast<BroadcastType::ROW>(dfb_in0, dfb_max, h, 0, dst0);
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
        for (std::uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_x_m_max_obj);
            copy_tile(dfb_x_m_max, h, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (h == Ht - 1) {
                copy_tile_init_with_dt(dfb_mask_obj);
                copy_tile(dfb_mask, 0, dst1);

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
            dfb_exps,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::col(Ht),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_exps,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::col(Ht),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
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

        for (std::uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_rows_init_with_dt(dfb_x_m_max_obj, dfb_recipsumexps_obj);
            sub_tiles_bcast<BroadcastType::ROW>(dfb_x_m_max, dfb_recipsumexps, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_out0_obj);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_rows_init_with_dt(dfb_exps_obj, dfb_recipsumexps_obj);
            mul_tiles_bcast_rows(dfb_exps, dfb_recipsumexps, h, 0, dst0);
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
