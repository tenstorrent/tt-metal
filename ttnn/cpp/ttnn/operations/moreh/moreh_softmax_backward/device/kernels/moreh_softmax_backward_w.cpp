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

    DataflowBuffer dfb_sum_obj(dfb::sum);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if (Wt == 1) {
            // apply mask
            mask_tile_to_dfb<dfb::dy, dfb::mask, dfb::dy_m_sum>(
                /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::dy_m_sum, dfb::scaler, dfb::sum>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // On this path the y*dy and sum buffers hold two partial sums instead; the second
            // names below are handle aliases for those same two FIFOs, not extra buffers.
            constexpr auto dfb_inter0_id = dfb::ydy;
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                dfb::dy,
                dfb::scaler,
                dfb_inter0_id,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            constexpr auto dfb_inter1_id = dfb::sum;
            mask_tile_to_dfb<dfb::dy, dfb::mask, dfb_inter1_id>(
                /*itile=*/Wt - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb_inter1_id, dfb::scaler, dfb::dy_m_sum>(
                compute_kernel_lib::ReduceInputBlockShape::single());

            add_tiles_to_dfb<dfb_inter0_id, dfb::dy_m_sum, dfb::sum>();
        }

        // dy - sum * exp(y)
        constexpr auto dfb_exp_id = dfb::ydy;  // the y * dy buffer, reused to hold exp(y)
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // exp(y)
            exp_tile_to_dfb<dfb::y, dfb_exp_id>(w, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_cols_to_dfb<dfb_exp_id, dfb::sum, dfb::dy_m_sum>(0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb::dy, dfb::dy_m_sum, dfb::dx>(w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Wt);
        dfb_dy_obj.pop_front(Wt);
#else
        // step 1, compute y * dy
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb::y, dfb::dy, dfb::mask, dfb::ydy>(
                    w, w, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb::y, dfb::dy, dfb::ydy>(w, w, /*pop0=*/0, /*pop1=*/0);
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
            sub_tiles_bcast_cols_to_dfb<dfb::dy, dfb::sum, dfb::dy_m_sum>(w, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>(w, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>(w, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Wt);
        dfb_y_obj.pop_front(Wt);
#endif
    }
}
