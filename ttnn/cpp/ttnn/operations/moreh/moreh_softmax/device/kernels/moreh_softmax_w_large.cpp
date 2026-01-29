// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_bcast_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Wt-1 tiles
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_in0, cb_bcast_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp,
                cb_bcast_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        // step 1
        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x)
            if (w == Wt - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_cb(
                    cb_tmp,
                    cb_mask,
                    cb_exps,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_cb(
                    cb_in0,
                    cb_mask,
                    cb_exps,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_cb(cb_tmp, cb_exps);
#else
                sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_cb(cb_tmp, cb_exps);
#endif
            }

            if (w == 0) {
                copy_tile_to_cb(cb_exps, cb_add);
            } else {
                add_tiles_to_cb(cb_add, cb_exps, cb_add);
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerBatch>(
                cb_add,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerBatch>(
                cb_add,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_cols_to_cb(cb_tmp, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(cb_tmp, cb_exps);

            mul_tiles_bcast_cols_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(cb_tmp, cb_exps);

            mul_tiles_bcast_cols_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        cb_pop_front(cb_recipsumexps, onetile);
        cb_pop_front(cb_max, onetile);
    }
}
