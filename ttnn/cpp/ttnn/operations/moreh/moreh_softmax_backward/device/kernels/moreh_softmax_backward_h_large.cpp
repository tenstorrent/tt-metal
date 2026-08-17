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
    DataflowBuffer dfb_add_obj(dfb::add);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                if (h == 0) {
                    mask_tile_to_cb(
                        dfb_dy_obj, dfb_mask_obj, dfb_add_obj, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);
                } else {
                    // The y*dy buffer under a second name; one FIFO, not an extra buffer.
                    auto& dfb_inter0_obj = dfb_ydy_obj;
                    mask_tile_to_cb(
                        dfb_dy_obj, dfb_mask_obj, dfb_inter0_obj, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);

                    add_tiles_to_cb(dfb_add_obj, dfb_inter0_obj, dfb_add_obj);
                }
            } else {
                if (h == 0) {
                    copy_tile_to_cb(dfb_dy_obj, dfb_add_obj);
                } else {
                    add_tiles_to_cb(dfb_add_obj, dfb_dy_obj, dfb_add_obj);
                }
            }
        }

        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb::add, dfb::scaler, dfb::sum>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        for (uint32_t h = 0; h < Ht; ++h) {
            // exp(y)
            auto& dfb_exp_obj = dfb_ydy_obj;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj, 0);

            // sum * exp(y)
            mul_tiles_bcast_rows_to_cb(dfb_exp_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
        }

        dfb_sum_obj.pop_front(onetile);
#else

        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    dfb_y_obj, dfb_dy_obj, dfb_mask_obj, dfb_ydy_obj, 0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                mul_tiles_to_cb(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj);
            }

            if (h == 0) {
                copy_tile_to_cb(dfb_ydy_obj, dfb_add_obj);
            } else {
                add_tiles_to_cb(dfb_add_obj, dfb_ydy_obj, dfb_add_obj);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb::add, dfb::scaler, dfb::sum>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_cb(dfb_dy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
