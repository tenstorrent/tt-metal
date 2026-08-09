// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto dfb_y_id = tt::CBIndex::c_0;
    constexpr auto dfb_dy_id = tt::CBIndex::c_1;
    constexpr auto dfb_dx_id = tt::CBIndex::c_16;

    constexpr auto dfb_ydy_id = tt::CBIndex::c_24;  // y * dy
    constexpr auto dfb_sum_id = tt::CBIndex::c_25;
    DataflowBuffer dfb_sum_obj(dfb_sum_id);
    constexpr auto dfb_dy_m_sum_id = tt::CBIndex::c_26;  // dy - sum

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t dim_size = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(dfb_dy_id, dfb_y_id, dfb_dx_id);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_dfb<dfb_dy_id, dfb_sum_id>();
            } else {
                add_tiles_to_dfb<dfb_sum_id, dfb_dy_id, dfb_sum_id>();
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            constexpr auto dfb_exp_id = tt::CBIndex::c_24;
            exp_tile_to_dfb<dfb_y_id, dfb_exp_id>();

            constexpr auto dfb_inter2_id = tt::CBIndex::c_26;
            mul_tiles_to_dfb<dfb_sum_id, dfb_exp_id, dfb_inter2_id>(0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb_dy_id, dfb_inter2_id, dfb_dx_id>();
        }
        dfb_sum_obj.pop_front(onetile);
#else
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_dfb<dfb_y_id, dfb_dy_id, dfb_ydy_id>();

            if (i == 0) {
                copy_tile_to_dfb<dfb_ydy_id, dfb_sum_id>();
            } else {
                add_tiles_to_dfb<dfb_sum_id, dfb_ydy_id, dfb_sum_id>();
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_dfb<dfb_dy_id, dfb_sum_id, dfb_dy_m_sum_id>(
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb_dy_m_sum_id, dfb_y_id, dfb_dx_id>();
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb_dy_m_sum_id, dfb_y_id, dfb_dx_id>();
#endif
        }
        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
