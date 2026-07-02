// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    CircularBuffer cb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    CircularBuffer cb_add_obj(cb_add);
    constexpr auto cb_max = tt::CBIndex::c_27;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    CircularBuffer cb_tmp_obj(cb_tmp);

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
                copy_tile_to_cb(cb_in0_obj, cb_max_obj);
            } else {
                cb_in0_obj.wait_front(onetile);
                cb_max_obj.wait_front(onetile);

                tile_regs_acquire();

                copy_tile_init_with_dt(cb_in0_obj);
                copy_tile(cb_in0, 0, dst0);

                copy_tile_init_with_dt(cb_max_obj);
                copy_tile(cb_max, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
                tile_regs_commit();

                cb_max_obj.pop_front(onetile);
                cb_max_obj.reserve_back(onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_max_obj);
                tile_regs_release();

                cb_max_obj.push_back(onetile);
                cb_in0_obj.pop_front(onetile);
            }
        }

        // compute exp(x - max(x))
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            sub_tiles_to_cb(cb_in0_obj, cb_max_obj, cb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(cb_tmp_obj, cb_exps_obj);
#else
            sub_tiles_to_cb(cb_in0_obj, cb_max_obj, cb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(cb_tmp_obj, cb_exps_obj);
#endif

            if (i == 0) {
                copy_tile_to_cb(cb_exps_obj, cb_add_obj);
            } else {
                add_tiles_to_cb(cb_add_obj, cb_exps_obj, cb_add_obj);
            }
        }

#ifdef LOG
        // compute log(sum)
        log_tile_to_cb(cb_add_obj, cb_recipsumexps_obj);
#else
        // compute 1/sum(exp(x))
        recip_tile_to_cb(cb_add_obj, cb_recipsumexps_obj);
#endif

        // step 3, compute final result
        cb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_to_cb(cb_in0_obj, cb_max_obj, cb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_to_cb(cb_tmp_obj, cb_recipsumexps_obj, cb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            sub_tiles_to_cb(cb_max_obj, cb_in0_obj, cb_tmp_obj, 0, 0, /*pop0=*/0, /*pop1=*/1);

            sub_tiles_to_cb(cb_tmp_obj, cb_recipsumexps_obj, cb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_to_cb(cb_in0_obj, cb_max_obj, cb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(cb_tmp_obj, cb_exps_obj);

            mul_tiles_to_cb(cb_exps_obj, cb_recipsumexps_obj, cb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_to_cb(cb_in0_obj, cb_max_obj, cb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(cb_tmp_obj, cb_exps_obj);

            mul_tiles_to_cb(cb_exps_obj, cb_recipsumexps_obj, cb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
