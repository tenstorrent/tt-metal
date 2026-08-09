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
    constexpr auto cb_in0 = dfb::in0;
    constexpr auto cb_mask = dfb::mask;
    constexpr auto cb_max_scaler = dfb::max_scaler;
    constexpr auto cb_sum_scaler = dfb::sum_scaler;
    constexpr auto cb_out0 = dfb::out0;
    constexpr auto cb_exps = dfb::exps;
    constexpr auto cb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = dfb::add;
    constexpr auto cb_max = dfb::max;
    DataflowBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = dfb::tmp;

    compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_out0);

    constexpr std::uint32_t onetile = 1;

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max (no accumulation, first call).
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in0, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile and continue reducing into cb_max via Accumulate.
            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(0, 0, /*pop0=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
        }

        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_cb<cb_tmp, cb_mask, cb_exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_cb<cb_in0, cb_mask, cb_exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_cb<cb_tmp, cb_exps>();
#else
                sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_cb<cb_tmp, cb_exps>();
#endif
            }

            if (w == 0) {
                copy_tile_to_cb<cb_exps, cb_add>();
            } else {
                add_tiles_to_cb<cb_add, cb_exps, cb_add>();
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
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
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
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
            sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_cols_to_cb<cb_tmp, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb<cb_tmp, cb_exps>();

            mul_tiles_bcast_cols_to_cb<cb_exps, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_cols_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb<cb_tmp, cb_exps>();

            mul_tiles_bcast_cols_to_cb<cb_exps, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
