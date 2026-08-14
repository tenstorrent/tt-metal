// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_mask_obj(dfb::mask);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    DataflowBuffer dfb_ydy_obj(dfb::ydy);  // y * dy
    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if (Wt == 1) {
            // apply mask
            mask_tile_to_cb(
                dfb_dy_obj, dfb_mask_obj, dfb_dy_m_sum_obj, /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::dy_m_sum, dfb::scaler, dfb::sum>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // On this path the y*dy and sum buffers hold two partial sums instead; the second
            // names below are handle aliases for those same two FIFOs, not extra buffers.
            constexpr auto dfb_inter0 = dfb::ydy;
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                dfb::dy,
                dfb::scaler,
                dfb_inter0,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            constexpr auto dfb_inter1 = dfb::sum;
            auto& dfb_inter1_obj = dfb_sum_obj;
            mask_tile_to_cb(
                dfb_dy_obj, dfb_mask_obj, dfb_inter1_obj, /*itile=*/Wt - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb_inter1, dfb::scaler, dfb::dy_m_sum>(
                compute_kernel_lib::ReduceInputBlockShape::single());

            auto& dfb_inter0_obj = dfb_ydy_obj;
            add_tiles_to_cb(dfb_inter0_obj, dfb_dy_m_sum_obj, dfb_sum_obj);
        }

        // dy - sum * exp(y)
        auto& dfb_exp_obj = dfb_ydy_obj;  // the y * dy buffer, reused to hold exp(y)

        for (uint32_t w = 0; w < Wt; w += onetile) {
            // exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj, w, /*dst=*/0, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_cols_to_cb(dfb_exp_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj, w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Wt);
        dfb_dy_obj.pop_front(Wt);
#else
        // step 1, compute y * dy
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    dfb_y_obj, dfb_dy_obj, dfb_mask_obj, dfb_ydy_obj, w, w, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                mul_tiles_to_cb(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj, w, w, /*pop0=*/0, /*pop1=*/0);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb::ydy,
            dfb::scaler,
            dfb::sum,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // dy - sum
            sub_tiles_bcast_cols_to_cb(dfb_dy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, w, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj, w, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj, w, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Wt);
        dfb_y_obj.pop_front(Wt);
#endif
    }
}
