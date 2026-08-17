// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto dfb_in0_id = dfb::in0;
    constexpr auto dfb_mask_id = dfb::mask;
    constexpr auto dfb_max_scaler_id = dfb::max_scaler;
    constexpr auto dfb_sum_scaler_id = dfb::sum_scaler;
    constexpr auto dfb_out0_id = dfb::out0;
    constexpr auto dfb_exps_id = dfb::exps;
    constexpr auto dfb_recipsumexps_id = dfb::recip_sum_exps;
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps_id);
    constexpr auto dfb_add_id = dfb::add;
    constexpr auto dfb_max_id = dfb::max;
    DataflowBuffer dfb_max_obj(dfb_max_id);
    constexpr auto dfb_tmp_id = dfb::tmp;

    compute_kernel_hw_startup(dfb_in0_id, dfb_max_scaler_id, dfb_out0_id);

    constexpr uint32_t onetile = 1;

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Ht == 1) {
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                ckl::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Ht-1 tiles
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_in0_id, dfb_max_scaler_id, dfb_max_id>(
                ckl::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(dfb_max_id, 1));
        }

        for (std::uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x - max(x))
            if (h == Ht - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_dfb<dfb_tmp_id, dfb_mask_id, dfb_exps_id>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_exps_id>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();
#else
                sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();
#endif
            }

            if (h == 0) {
                copy_tile_to_dfb<dfb_exps_id, dfb_add_id>();
            } else {
                add_tiles_to_dfb<dfb_add_id, dfb_exps_id, dfb_add_id>();
            }
        }

#ifdef LOG
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_add_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb_add_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result
        for (std::uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_rows_to_dfb<dfb_tmp_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();

            mul_tiles_bcast_rows_to_dfb<dfb_exps_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();

            mul_tiles_bcast_rows_to_dfb<dfb_exps_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
