// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    DataflowBuffer dfb_y_obj(cb_y);
    constexpr auto cb_dy = tt::CBIndex::c_1;
    DataflowBuffer dfb_dy_obj(cb_dy);
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;
    DataflowBuffer dfb_mask_obj(cb_mask);
    constexpr auto cb_dx = tt::CBIndex::c_16;
    DataflowBuffer dfb_dx_obj(cb_dx);

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    DataflowBuffer dfb_ydy_obj(cb_ydy);
    constexpr auto cb_sum = tt::CBIndex::c_25;
    DataflowBuffer dfb_sum_obj(cb_sum);
    constexpr auto cb_inter2 = tt::CBIndex::c_26;
    DataflowBuffer dfb_inter2_obj(cb_inter2);

    compute_kernel_hw_startup(cb_y, cb_bcast_scaler, cb_dx);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if (Ht == 1) {
            // apply mask
            mask_tile_to_cb(dfb_dy_obj, dfb_mask_obj, dfb_inter2_obj, /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_inter2, cb_bcast_scaler, cb_sum>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            constexpr auto cb_inter0 = tt::CBIndex::c_24;
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                cb_dy,
                cb_bcast_scaler,
                cb_inter0,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            constexpr auto cb_inter1 = tt::CBIndex::c_25;
            DataflowBuffer dfb_inter1_obj(cb_inter1);
            mask_tile_to_cb(
                dfb_dy_obj, dfb_mask_obj, dfb_inter1_obj, /*itile=*/Ht - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_inter1, cb_bcast_scaler, cb_inter2>(
                compute_kernel_lib::ReduceInputBlockShape::single());

            DataflowBuffer dfb_inter0_obj(cb_inter0);
            add_tiles_to_cb(dfb_inter0_obj, dfb_inter2_obj, dfb_sum_obj);
        }

        // dy - sum * exp(y)
        constexpr auto cb_exp = tt::CBIndex::c_24;  // y * dy
        DataflowBuffer dfb_exp_obj(cb_exp);

        for (uint32_t w = 0; w < Ht; w += onetile) {
            // exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj, w, /*dst=*/0, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_rows_to_cb(dfb_exp_obj, dfb_sum_obj, dfb_inter2_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_inter2_obj, dfb_dx_obj, w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Ht);
        dfb_dy_obj.pop_front(Ht);
#else
        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    dfb_y_obj, dfb_dy_obj, dfb_mask_obj, dfb_ydy_obj, h, h, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                mul_tiles_to_cb(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj, h, h, /*pop0=*/0, /*pop1=*/0);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_ydy,
            cb_bcast_scaler,
            cb_sum,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::col(Ht));

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_cb(dfb_dy_obj, dfb_sum_obj, dfb_inter2_obj, h, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_y_obj, dfb_inter2_obj, dfb_dx_obj, h, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_y_obj, dfb_inter2_obj, dfb_dx_obj, h, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Ht);
        dfb_y_obj.pop_front(Ht);
#endif
    }
}
