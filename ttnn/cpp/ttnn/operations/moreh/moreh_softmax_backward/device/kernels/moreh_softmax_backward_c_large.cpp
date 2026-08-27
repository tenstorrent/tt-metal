// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    DataflowBuffer dfb_ydy_obj(dfb::ydy);  // y * dy
    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);  // dy - sum

    uint32_t N = get_arg(args::N);
    uint32_t dim_size = get_arg(args::dim_size);

    compute_kernel_hw_startup(dfb::dy, dfb::y, dfb::dx);

    constexpr int dst0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb(dfb_dy_obj, dfb_sum_obj);
            } else {
                add_tiles_to_cb(dfb_sum_obj, dfb_dy_obj, dfb_sum_obj);
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // exp(y)
            auto& dfb_exp_obj = dfb_ydy_obj;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj);

            // sum * exp(y)
            mul_tiles_to_cb(dfb_sum_obj, dfb_exp_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
        }
        dfb_sum_obj.pop_front(onetile);
#else
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj);

            if (i == 0) {
                copy_tile_to_cb(dfb_ydy_obj, dfb_sum_obj);
            } else {
                add_tiles_to_cb(dfb_sum_obj, dfb_ydy_obj, dfb_sum_obj);
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_cb(
                dfb_dy_obj,
                dfb_sum_obj,
                dfb_dy_m_sum_obj,
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_dy_m_sum_obj, dfb_y_obj, dfb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_dy_m_sum_obj, dfb_y_obj, dfb_dx_obj);
#endif
        }
        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
