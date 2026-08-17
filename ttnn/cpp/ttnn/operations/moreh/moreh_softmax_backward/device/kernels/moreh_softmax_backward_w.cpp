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

    constexpr auto dfb_y_id = dfb::y;
    DataflowBuffer dfb_y_obj(dfb_y_id);
    constexpr auto dfb_dy_id = dfb::dy;
    DataflowBuffer dfb_dy_obj(dfb_dy_id);
    constexpr auto dfb_scaler_id = dfb::scaler;
    constexpr auto dfb_mask_id = dfb::mask;
    constexpr auto dfb_dx_id = dfb::dx;

    constexpr auto dfb_ydy_id = dfb::ydy;  // y * dy
    constexpr auto dfb_sum_id = dfb::sum;
    DataflowBuffer dfb_sum_obj(dfb_sum_id);
    constexpr auto dfb_dy_m_sum_id = dfb::dy_m_sum;

    compute_kernel_hw_startup(dfb_y_id, dfb_scaler_id, dfb_dx_id);

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if (Wt == 1) {
            // apply mask
            mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_dy_m_sum_id>(
                /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb_dy_m_sum_id, dfb_scaler_id, dfb_sum_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            constexpr auto dfb_inter0_id = dfb_ydy_id;
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                dfb_dy_id,
                dfb_scaler_id,
                dfb_inter0_id,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            constexpr auto dfb_inter1_id = dfb_sum_id;
            mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_inter1_id>(
                /*itile=*/Wt - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb_inter1_id, dfb_scaler_id, dfb_dy_m_sum_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());

            add_tiles_to_dfb<dfb_inter0_id, dfb_dy_m_sum_id, dfb_sum_id>();
        }

        // dy - sum * exp(y)
        constexpr auto dfb_exp_id = dfb_ydy_id;  // y * dy
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // exp(y)
            exp_tile_to_dfb<dfb_y_id, dfb_exp_id>(w, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_cols_to_dfb<dfb_exp_id, dfb_sum_id, dfb_dy_m_sum_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb_dy_id, dfb_dy_m_sum_id, dfb_dx_id>(w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Wt);
        dfb_dy_obj.pop_front(Wt);
#else
        // step 1, compute y * dy
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb_y_id, dfb_dy_id, dfb_mask_id, dfb_ydy_id>(
                    w, w, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb_y_id, dfb_dy_id, dfb_ydy_id>(w, w, /*pop0=*/0, /*pop1=*/0);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_ydy_id,
            dfb_scaler_id,
            dfb_sum_id,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // dy - sum
            sub_tiles_bcast_cols_to_dfb<dfb_dy_id, dfb_sum_id, dfb_dy_m_sum_id>(w, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb_y_id, dfb_dy_m_sum_id, dfb_dx_id>(w, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb_y_id, dfb_dy_m_sum_id, dfb_dx_id>(w, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Wt);
        dfb_y_obj.pop_front(Wt);
#endif
    }
}
