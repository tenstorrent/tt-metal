// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef LOG
constexpr auto reduce_input = dfb::dy;
#else
constexpr auto reduce_input = dfb::ydy;
#endif
using ReduceCall =
    ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallArgs<0>, reduce_input, dfb::scaler, dfb::sum>;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    DataflowBuffer dfb_ydy_obj(dfb::ydy);  // y * dy
    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // The plan includes the partial last tile and keeps dy for the output pass.
        compute_kernel_lib::reduce<ReduceCall>();

        // dy - sum * exp(y)
        auto& dfb_exp_obj = dfb_ydy_obj;  // the y * dy buffer, reused to hold exp(y)

        for (uint32_t w = 0; w < Ht; w += onetile) {
            // exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj, w, /*dst=*/0, /*pop=*/0);

            // sum * exp(y)
            mul_tiles_bcast_rows_to_cb(dfb_exp_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj, w, 0, /*pop0=*/0, /*pop1=*/1);
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_y_obj.pop_front(Ht);
        dfb_dy_obj.pop_front(Ht);
#else
        for (uint32_t h = 0; h < Ht; ++h) {
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_obj, dfb_ydy_obj, h, h, /*pop0=*/0, /*pop1=*/0);
        }
        compute_kernel_lib::reduce<ReduceCall>();

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_cb(dfb_dy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, h, 0, /*pop0=*/0, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj, h, 0, /*pop0=*/0, /*pop1=*/1);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj, h, 0, /*pop0=*/0, /*pop1=*/1);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(Ht);
        dfb_y_obj.pop_front(Ht);
#endif
    }
    DataflowBuffer(dfb::scaler).pop_front(ReduceCall::auxiliary_tile_count);
}
