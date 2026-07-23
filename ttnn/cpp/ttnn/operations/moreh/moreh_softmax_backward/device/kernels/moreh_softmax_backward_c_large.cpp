// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    DataflowBuffer cb_sum_obj(cb_sum);
    constexpr auto cb_dy_m_sum = tt::CBIndex::c_26;  // dy - sum

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y, cb_dx);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb<cb_dy, cb_sum>();
            } else {
                add_tiles_to_cb<cb_sum, cb_dy, cb_sum>();
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            constexpr auto cb_exp = tt::CBIndex::c_24;
            exp_tile_to_cb<cb_y, cb_exp>();

            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            mul_tiles_to_cb<cb_sum, cb_exp, cb_inter2>(0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_cb<cb_dy, cb_inter2, cb_dx>();
        }
        cb_sum_obj.pop_front(onetile);
#else
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_cb<cb_y, cb_dy, cb_ydy>();

            if (i == 0) {
                copy_tile_to_cb<cb_ydy, cb_sum>();
            } else {
                add_tiles_to_cb<cb_sum, cb_ydy, cb_sum>();
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_cb<cb_dy, cb_sum, cb_dy_m_sum>(
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb<cb_dy_m_sum, cb_y, cb_dx>();
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb<cb_dy_m_sum, cb_y, cb_dx>();
#endif
        }
        cb_sum_obj.pop_front(onetile);
#endif
    }
}
