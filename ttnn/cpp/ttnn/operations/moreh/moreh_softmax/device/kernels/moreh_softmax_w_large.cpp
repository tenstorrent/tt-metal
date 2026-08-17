// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: with fp32_dest_acc_en this kernel must be built at -O3 (the program factory sets the compute
// KernelSpec opt_level to O3, matching legacy). At -O2 GCC fails to constant-fold the LLK addrmod SETC16
// inline-asm "n" immediate in this larger fp32 TU and JIT aborts with "impossible constraint in 'asm'".
// At O3 it folds and no source workaround is needed.

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

    constexpr std::uint32_t onetile = 1;

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into dfb_max_id (no accumulation, first call).
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_in0_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile and continue reducing into dfb_max_id via Accumulate.
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb_max_id, /*iter=*/1));
        }

        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

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
                sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();
#else
                sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();
#endif
            }

            if (w == 0) {
                copy_tile_to_dfb<dfb_exps_id, dfb_add_id>();
            } else {
                add_tiles_to_dfb<dfb_add_id, dfb_exps_id, dfb_add_id>();
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
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
        // compute 1/sum(exp(x)) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
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
        for (std::uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_cols_to_dfb<dfb_tmp_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();

            mul_tiles_bcast_cols_to_dfb<dfb_exps_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_cols_to_dfb<dfb_in0_id, dfb_max_id, dfb_tmp_id>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb_tmp_id, dfb_exps_id>();

            mul_tiles_bcast_cols_to_dfb<dfb_exps_id, dfb_recipsumexps_id, dfb_out0_id>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
