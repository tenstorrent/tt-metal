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

    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_mask_obj(dfb::mask);
    DataflowBuffer dfb_ydy_obj(dfb::ydy);
    DataflowBuffer dfb_add_obj(dfb::add);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                if (w == 0) {
                    mask_tile_to_dfb<dfb::dy, dfb::mask, dfb::add>(
                        dfb_dy_obj,
                        dfb_mask_obj,
                        dfb_add_obj,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);
                } else {
                    // The y*dy buffer under a second name; one FIFO, not an extra buffer.
                    constexpr auto dfb_inter0_id = dfb::ydy;
                    mask_tile_to_dfb<dfb::dy, dfb::mask, dfb_inter0_id>(
                        dfb_dy_obj,
                        dfb_mask_obj,
                        dfb_ydy_obj,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);

                    add_tiles_to_dfb<dfb::add, dfb_inter0_id, dfb::add>(dfb_add_obj, dfb_ydy_obj, dfb_add_obj);
                }
            } else {
                if (w == 0) {
                    copy_tile_to_dfb<dfb::dy, dfb::add>(dfb_dy_obj, dfb_add_obj);
                } else {
                    add_tiles_to_dfb<dfb::add, dfb::dy, dfb::add>(dfb_add_obj, dfb_dy_obj, dfb_add_obj);
                }
            }
        }

        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::add, dfb::scaler, dfb::sum>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t w = 0; w < Wt; w += onetile) {
            constexpr auto dfb_exp_id = dfb::ydy;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_dfb<dfb::y, dfb_exp_id>(dfb_y_obj, dfb_ydy_obj);
            // sum * exp(y)
            mul_tiles_bcast_cols_to_dfb<dfb_exp_id, dfb::sum, dfb::dy_m_sum>(
                dfb_ydy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb::dy, dfb::dy_m_sum, dfb::dx>(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
        }

        dfb_sum_obj.pop_front(onetile);
#else
        // step 1, compute y * dy
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb::y, dfb::dy, dfb::mask, dfb::ydy>(
                    dfb_y_obj, dfb_dy_obj, dfb_mask_obj, dfb_ydy_obj, 0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb::y, dfb::dy, dfb::ydy>(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj);
            }

            if (w == 0) {
                copy_tile_to_dfb<dfb::ydy, dfb::add>(dfb_ydy_obj, dfb_add_obj);
            } else {
                add_tiles_to_dfb<dfb::add, dfb::ydy, dfb::add>(dfb_add_obj, dfb_ydy_obj, dfb_add_obj);
            }
        }

        // step 2, compute sum(y * dy)
        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::add, dfb::scaler, dfb::sum>(
            ckl::ReduceInputBlockShape::single());

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // dy - sum
            sub_tiles_bcast_cols_to_dfb<dfb::dy, dfb::sum, dfb::dy_m_sum>(
                dfb_dy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
