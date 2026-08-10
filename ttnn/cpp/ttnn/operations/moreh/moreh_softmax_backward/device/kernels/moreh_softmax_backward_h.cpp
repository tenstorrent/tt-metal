// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto dfb_y_id = tt::CBIndex::c_0;
    DataflowBuffer dfb_y_obj(dfb_y_id);
    constexpr auto dfb_dy_id = tt::CBIndex::c_1;
    DataflowBuffer dfb_dy_obj(dfb_dy_id);
    constexpr auto dfb_bcast_scaler_id = tt::CBIndex::c_2;
    constexpr auto dfb_mask_id = tt::CBIndex::c_3;
    constexpr auto dfb_dx_id = tt::CBIndex::c_16;

    constexpr auto dfb_ydy_id = tt::CBIndex::c_24;  // y * dy
    constexpr auto dfb_sum_id = tt::CBIndex::c_25;
    DataflowBuffer dfb_sum_obj(dfb_sum_id);
    constexpr auto dfb_inter2_id = tt::CBIndex::c_26;

    compute_kernel_hw_startup(dfb_y_id, dfb_bcast_scaler_id, dfb_dx_id);

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if constexpr (Ht == 1) {
            // apply mask
            mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_inter2_id>(
                /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb_inter2_id, dfb_bcast_scaler_id, dfb_sum_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            constexpr auto dfb_inter0_id = tt::CBIndex::c_24;
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                dfb_dy_id,
                dfb_bcast_scaler_id,
                dfb_inter0_id,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            constexpr auto dfb_inter1_id = tt::CBIndex::c_25;
            mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_inter1_id>(
                /*itile=*/Ht - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            constexpr auto dfb_inter2_id = tt::CBIndex::c_26;
            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb_inter1_id, dfb_bcast_scaler_id, dfb_inter2_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());

            add_tiles_to_dfb<dfb_inter0_id, dfb_inter2_id, dfb_sum_id>();
        }

        // dy - sum * exp(y)
        constexpr auto dfb_exp_id = tt::CBIndex::c_24;  // y * dy
        for (uint32_t w = 0; w < Ht; w += onetile) {
            // exp(y)
            exp_tile_to_dfb<dfb_y_id, dfb_exp_id>(w, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_rows_to_dfb<dfb_exp_id, dfb_sum_id, dfb_inter2_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb_dy_id, dfb_inter2_id, dfb_dx_id>(w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Ht);
        dfb_dy_obj.pop_front(Ht);
#else
        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb_y_id, dfb_dy_id, dfb_mask_id, dfb_ydy_id>(
                    h, h, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb_y_id, dfb_dy_id, dfb_ydy_id>(h, h, /*pop0=*/0, /*pop1=*/0);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_ydy_id,
            dfb_bcast_scaler_id,
            dfb_sum_id,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::col(Ht));

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_dfb<dfb_dy_id, dfb_sum_id, dfb_inter2_id>(h, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb_y_id, dfb_inter2_id, dfb_dx_id>(h, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb_y_id, dfb_inter2_id, dfb_dx_id>(h, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Ht);
        dfb_y_obj.pop_front(Ht);
#endif
    }
}
