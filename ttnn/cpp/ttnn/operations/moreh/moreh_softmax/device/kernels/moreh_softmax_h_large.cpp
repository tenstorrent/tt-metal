// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    DataflowBuffer dfb_recipsumexps_obj(dfb::recip_sum_exps);
    DataflowBuffer dfb_max_obj(dfb::max);

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::out0);

    constexpr uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Ht loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Ht == 1) {
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::tmp, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Ht-1 tiles
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::in0, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::tmp, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(dfb::max, 1));  // iteration=1, reload from dfb::max
        }

        for (std::uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x - max(x))
            if (h == Ht - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_dfb<dfb::tmp, dfb::mask, dfb::exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_dfb<dfb::tmp, dfb::exps>();
#else
                sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_dfb<dfb::tmp, dfb::exps>();
#endif
            }

            if (h == 0) {
                copy_tile_to_dfb<dfb::exps, dfb::add>();
            } else {
                add_tiles_to_dfb<dfb::add, dfb::exps, dfb::add>();
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
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
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_rows_to_dfb<dfb::tmp, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_bcast_rows_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_bcast_rows_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
