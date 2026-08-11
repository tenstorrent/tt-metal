// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto dfb_in0 = dfb::in0;
    DataflowBuffer dfb_in0_obj(dfb_in0);
    constexpr auto dfb_mask = dfb::mask;
    DataflowBuffer dfb_mask_obj(dfb_mask);
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_out0 = dfb::out0;
    DataflowBuffer dfb_out0_obj(dfb_out0);
    constexpr auto dfb_exps = dfb::exps;
    DataflowBuffer dfb_exps_obj(dfb_exps);
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    constexpr auto dfb_add = dfb::add;
    DataflowBuffer dfb_add_obj(dfb_add);
    constexpr auto dfb_max = dfb::max;
    DataflowBuffer dfb_max_obj(dfb_max);
    constexpr auto dfb_tmp = dfb::tmp;
    DataflowBuffer dfb_tmp_obj(dfb_tmp);

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_out0);

    constexpr std::uint32_t onetile = 1;
    constexpr int dst0 = 0;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Ht loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    std::uint32_t N = get_arg(args::N);
    std::uint32_t Ht = get_arg(args::Ht);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Ht == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp, dfb_max_scaler, dfb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Ht-1 tiles
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_in0, dfb_max_scaler, dfb_max>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp, dfb_max_scaler, dfb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb_max, 1));  // iteration=1, reload from dfb_max
        }

        for (std::uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x - max(x))
            if (h == Ht - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

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
                sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#else
                sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#endif
            }

            if (h == 0) {
                copy_tile_to_cb(dfb_exps_obj, dfb_add_obj);
            } else {
                add_tiles_to_cb(dfb_add_obj, dfb_exps_obj, dfb_add_obj);
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_add,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_add,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result
        for (std::uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_rows_to_cb(dfb_tmp_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_rows_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_rows_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
