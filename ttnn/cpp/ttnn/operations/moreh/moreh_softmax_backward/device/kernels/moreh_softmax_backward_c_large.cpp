// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    CircularBuffer cb_y_obj(cb_y);
    constexpr auto cb_dy = tt::CBIndex::c_1;
    CircularBuffer cb_dy_obj(cb_dy);
    constexpr auto cb_dx = tt::CBIndex::c_16;
    CircularBuffer cb_dx_obj(cb_dx);

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    CircularBuffer cb_ydy_obj(cb_ydy);
    constexpr auto cb_sum = tt::CBIndex::c_25;
    CircularBuffer cb_sum_obj(cb_sum);
    constexpr auto cb_dy_m_sum = tt::CBIndex::c_26;  // dy - sum
    CircularBuffer cb_dy_m_sum_obj(cb_dy_m_sum);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y, cb_dx);

    constexpr int dst0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb(cb_dy_obj, cb_sum_obj);
            } else {
                add_tiles_to_cb(cb_sum_obj, cb_dy_obj, cb_sum_obj);
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // exp(y)
            constexpr auto cb_exp = tt::CBIndex::c_24;
            CircularBuffer cb_exp_obj(cb_exp);
            exp_tile_to_cb(cb_y_obj, cb_exp_obj);

            // sum * exp(y)
            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            CircularBuffer cb_inter2_obj(cb_inter2);
            mul_tiles_to_cb(cb_sum_obj, cb_exp_obj, cb_inter2_obj, 0, 0, /*pop0=*/0, /*pop1=*/1);

            // dy - sum * exp(y)
            sub_tiles_to_cb(cb_dy_obj, cb_inter2_obj, cb_dx_obj);
        }
        cb_sum_obj.pop_front(onetile);
#else
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            mul_tiles_to_cb(cb_y_obj, cb_dy_obj, cb_ydy_obj);

            if (i == 0) {
                copy_tile_to_cb(cb_ydy_obj, cb_sum_obj);
            } else {
                add_tiles_to_cb(cb_sum_obj, cb_ydy_obj, cb_sum_obj);
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            sub_tiles_to_cb(
                cb_dy_obj,
                cb_sum_obj,
                cb_dy_m_sum_obj,
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(cb_dy_m_sum_obj, cb_y_obj, cb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(cb_dy_m_sum_obj, cb_y_obj, cb_dx_obj);
#endif
        }
        cb_sum_obj.pop_front(onetile);
#endif
    }
}
