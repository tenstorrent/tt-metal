// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto dfb_y_id = dfb::y;
    constexpr auto dfb_dy_id = dfb::dy;
    constexpr auto dfb_scaler_id = dfb::scaler;
    constexpr auto dfb_mask_id = dfb::mask;
    constexpr auto dfb_dx_id = dfb::dx;

    constexpr auto dfb_ydy_id = dfb::ydy;  // y * dy
    constexpr auto dfb_sum_id = dfb::sum;
    DataflowBuffer dfb_sum_obj(dfb_sum_id);
    constexpr auto dfb_dy_m_sum_id = dfb::dy_m_sum;
    constexpr auto dfb_add_id = dfb::add;

    compute_kernel_hw_startup(dfb_y_id, dfb_scaler_id, dfb_dx_id);

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                if (h == 0) {
                    mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_add_id>(
                        /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);
                } else {
                    constexpr auto dfb_inter0_id = dfb_ydy_id;
                    mask_tile_to_dfb<dfb_dy_id, dfb_mask_id, dfb_inter0_id>(
                        /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);

                    add_tiles_to_dfb<dfb_add_id, dfb_inter0_id, dfb_add_id>();
                }
            } else {
                if (h == 0) {
                    copy_tile_to_dfb<dfb_dy_id, dfb_add_id>();
                } else {
                    add_tiles_to_dfb<dfb_add_id, dfb_dy_id, dfb_add_id>();
                }
            }
        }

        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb_add_id, dfb_scaler_id, dfb_sum_id>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t h = 0; h < Ht; ++h) {
            constexpr auto dfb_exp_id = dfb_ydy_id;
            exp_tile_to_dfb<dfb_y_id, dfb_exp_id>();

            // sum * exp(y)
            mul_tiles_bcast_rows_to_dfb<dfb_exp_id, dfb_sum_id, dfb_dy_m_sum_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb_dy_id, dfb_dy_m_sum_id, dfb_dx_id>();
        }

        dfb_sum_obj.pop_front(onetile);
#else

        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb_y_id, dfb_dy_id, dfb_mask_id, dfb_ydy_id>(
                    0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb_y_id, dfb_dy_id, dfb_ydy_id>();
            }

            if (h == 0) {
                copy_tile_to_dfb<dfb_ydy_id, dfb_add_id>();
            } else {
                add_tiles_to_dfb<dfb_add_id, dfb_ydy_id, dfb_add_id>();
            }
        }

        // step 2, compute sum(y * dy)
        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb_add_id, dfb_scaler_id, dfb_sum_id>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_dfb<dfb_dy_id, dfb_sum_id, dfb_dy_m_sum_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb_y_id, dfb_dy_m_sum_id, dfb_dx_id>();
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb_y_id, dfb_dy_m_sum_id, dfb_dx_id>();
#endif
        }

        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
