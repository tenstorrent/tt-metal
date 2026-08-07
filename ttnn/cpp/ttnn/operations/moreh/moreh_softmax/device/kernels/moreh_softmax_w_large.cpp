// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    DataflowBuffer dfb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    DataflowBuffer dfb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    DataflowBuffer dfb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    DataflowBuffer dfb_add_obj(cb_add);
    constexpr auto cb_max = tt::CBIndex::c_27;
    DataflowBuffer dfb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    DataflowBuffer dfb_tmp_obj(cb_tmp);

    compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_out0);

    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max (no accumulation, first call).
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in0, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile and continue reducing into cb_max via Accumulate.
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
        }

        // step 1
        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x)
            if (w == Wt - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_cb(
                    dfb_tmp_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_cb(
                    dfb_in0_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#else
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#endif
            }

            if (w == 0) {
                copy_tile_to_cb(dfb_exps_obj, dfb_add_obj);
            } else {
                add_tiles_to_cb(dfb_add_obj, dfb_exps_obj, dfb_add_obj);
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
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
            sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_cols_to_cb(dfb_tmp_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_cols_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_cols_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
