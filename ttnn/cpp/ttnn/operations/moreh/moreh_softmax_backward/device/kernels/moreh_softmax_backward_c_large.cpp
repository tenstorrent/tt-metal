// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_ydy_obj(dfb::ydy);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t dim_size = get_arg(args::dim_size);

    compute_kernel_hw_startup(dfb::dy, dfb::y, dfb::dx);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_dfb<dfb::dy, dfb::sum>(dfb_dy_obj, dfb_sum_obj);
            } else {
                add_tiles_to_dfb<dfb::sum, dfb::dy, dfb::sum>(dfb_sum_obj, dfb_dy_obj, dfb_sum_obj);
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            constexpr auto dfb_exp_id = dfb::ydy;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_dfb<dfb::y, dfb_exp_id>(dfb_y_obj, dfb_ydy_obj);

            mul_tiles_to_dfb<dfb::sum, dfb_exp_id, dfb::dy_m_sum>(
                dfb_sum_obj, dfb_ydy_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb::dy, dfb::dy_m_sum, dfb::dx>(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
        }
        dfb_sum_obj.pop_front(onetile);
#else
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_dfb<dfb::y, dfb::dy, dfb::ydy>(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj);

            if (i == 0) {
                copy_tile_to_dfb<dfb::ydy, dfb::sum>(dfb_ydy_obj, dfb_sum_obj);
            } else {
                add_tiles_to_dfb<dfb::sum, dfb::ydy, dfb::sum>(dfb_sum_obj, dfb_ydy_obj, dfb_sum_obj);
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_dfb<dfb::dy, dfb::sum, dfb::dy_m_sum>(
                dfb_dy_obj,
                dfb_sum_obj,
                dfb_dy_m_sum_obj,
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb::dy_m_sum, dfb::y, dfb::dx>(dfb_dy_m_sum_obj, dfb_y_obj, dfb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb::dy_m_sum, dfb::y, dfb::dx>(dfb_dy_m_sum_obj, dfb_y_obj, dfb_dx_obj);
#endif
        }
        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
