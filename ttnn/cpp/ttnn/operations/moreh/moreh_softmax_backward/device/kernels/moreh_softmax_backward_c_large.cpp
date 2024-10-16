// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_ydy = tt::CB::c_intermed0;  // y * dy
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_dy_m_sum = tt::CB::c_intermed2;  // dy - sum

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y);

    constexpr int dst0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb(cb_dy, cb_sum);
            } else {
                add_tiles_to_cb(cb_sum, cb_dy, cb_sum);
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // exp(y)
            constexpr auto cb_exp = tt::CB::c_intermed0;
            exp_tile_to_cb(cb_y, cb_exp);

            // sum * exp(y)
            constexpr auto cb_inter2 = tt::CB::c_intermed2;
            mul_tiles_to_cb(cb_sum, cb_exp, cb_inter2, 0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_cb(cb_dy, cb_inter2, cb_dx);
        }
        cb_pop_front(cb_sum, onetile);
#else
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_cb(cb_y, cb_dy, cb_ydy);

            if (i == 0) {
                copy_tile_to_cb(cb_ydy, cb_sum);
            } else {
                add_tiles_to_cb(cb_sum, cb_ydy, cb_sum);
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_cb(cb_dy,
                            cb_sum,
                            cb_dy_m_sum,
                            /*itile0=*/0,
                            /*itile1=*/0,
                            /*pop0=*/1,
                            /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(cb_dy_m_sum, cb_y, cb_dx);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(cb_dy_m_sum, cb_y, cb_dx);
#endif
        }
        cb_pop_front(cb_sum, onetile);
#endif
    }
}
}  // namespace NAMESPACE
