// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps, cb_out0);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb(cb_in0, cb_max);
            } else {
                cb_wait_front(cb_in0, onetile);
                cb_wait_front(cb_max, onetile);

                tile_regs_acquire();

                copy_tile_init_with_dt(cb_in0);
                copy_tile(cb_in0, 0, dst0);

                copy_tile_init_with_dt(cb_max);
                copy_tile(cb_max, 0, dst1);

                max_tile_init();
                max_tile(dst0, dst1);
                tile_regs_commit();

                cb_pop_front(cb_max, onetile);
                cb_reserve_back(cb_max, onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_max);
                tile_regs_release();

                cb_push_back(cb_max, onetile);
                cb_pop_front(cb_in0, onetile);
            }
        }

        // compute exp(x - max(x))
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            sub_tiles_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(cb_tmp, cb_exps);
#else
            sub_tiles_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(cb_tmp, cb_exps);
#endif

            if (i == 0) {
                copy_tile_to_cb(cb_exps, cb_add);
            } else {
                add_tiles_to_cb(cb_add, cb_exps, cb_add);
            }
        }

#ifdef LOG
        // compute log(sum)
        log_tile_to_cb(cb_add, cb_recipsumexps);
#else
        // compute 1/sum(exp(x))
        recip_tile_to_cb(cb_add, cb_recipsumexps);
#endif

        // step 3, compute final result
        cb_wait_front(cb_recipsumexps, onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_to_cb(cb_tmp, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            sub_tiles_to_cb(cb_max, cb_in0, cb_tmp, 0, 0, /*pop0=*/0, /*pop1=*/1);

            sub_tiles_to_cb(cb_tmp, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(cb_tmp, cb_exps);

            mul_tiles_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(cb_tmp, cb_exps);

            mul_tiles_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        cb_pop_front(cb_recipsumexps, onetile);
        cb_pop_front(cb_max, onetile);
    }
}
}  // namespace NAMESPACE
