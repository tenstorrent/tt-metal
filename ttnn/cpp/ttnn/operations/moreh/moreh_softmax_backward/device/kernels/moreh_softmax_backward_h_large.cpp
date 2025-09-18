// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    constexpr auto cb_inter2 = tt::CBIndex::c_26;
    constexpr auto cb_add = tt::CBIndex::c_27;

    binary_op_init_common(cb_y, cb_bcast_scaler, cb_dx);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                if (h == 0) {
                    mask_tile_to_cb(cb_dy, cb_mask, cb_add, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);
                } else {
                    constexpr auto cb_inter0 = tt::CBIndex::c_24;
                    mask_tile_to_cb(cb_dy, cb_mask, cb_inter0, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);

                    add_tiles_to_cb(cb_add, cb_inter0, cb_add);
                }
            } else {
                if (h == 0) {
                    copy_tile_to_cb(cb_dy, cb_add);
                } else {
                    add_tiles_to_cb(cb_add, cb_dy, cb_add);
                }
            }
        }

        reduce_tile_to_cb<REDUCE_OP, REDUCE_DIM>(cb_add, cb_bcast_scaler, cb_sum, 1, /*pop0=*/1, /*pop1=*/0);

        for (uint32_t h = 0; h < Ht; ++h) {
            // exp(y)
            constexpr auto cb_exp = tt::CBIndex::c_24;
            exp_tile_to_cb(cb_y, cb_exp, 0);

            // sum * exp(y)
            mul_tiles_bcast_rows_to_cb(cb_exp, cb_sum, cb_inter2, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(cb_dy, cb_inter2, cb_dx);
        }

        cb_pop_front(cb_sum, onetile);
#else

        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    cb_y, cb_dy, cb_mask, cb_ydy, 0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                mul_tiles_to_cb(cb_y, cb_dy, cb_ydy);
            }

            if (h == 0) {
                copy_tile_to_cb(cb_ydy, cb_add);
            } else {
                add_tiles_to_cb(cb_add, cb_ydy, cb_add);
            }
        }

        // step 2, compute sum(y * dy)
        reduce_tile_to_cb<REDUCE_OP, REDUCE_DIM>(cb_add, cb_bcast_scaler, cb_sum, /*size=*/1, /*pop0=*/1, /*pop1=*/0);

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_cb(cb_dy, cb_sum, cb_inter2, 0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(cb_y, cb_inter2, cb_dx);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx);
#endif
        }

        cb_pop_front(cb_sum, onetile);
#endif
    }
}
}  // namespace NAMESPACE
