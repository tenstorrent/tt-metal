// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    DataflowBuffer dfb_recipsumexps_obj(dfb::recip_sum_exps);
    DataflowBuffer dfb_max_obj(dfb::max);

    constexpr uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-dim_size loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    uint32_t N = get_arg(args::N);
    uint32_t dim_size = get_arg(args::dim_size);

    compute_kernel_hw_startup(dfb::in0, dfb::exps, dfb::out0);

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_dfb<dfb::in0, dfb::max>();
            } else {
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    ckl::input(dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb::max, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(dfb::max, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::tiles(onetile));
            }
        }

        // compute exp(x - max(x))
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            sub_tiles_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb::tmp, dfb::exps>();
#else
            sub_tiles_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb::tmp, dfb::exps>();
#endif

            if (i == 0) {
                copy_tile_to_dfb<dfb::exps, dfb::add>();
            } else {
                add_tiles_to_dfb<dfb::add, dfb::exps, dfb::add>();
            }
        }

#ifdef LOG
        // compute log(sum)
        log_tile_to_dfb<dfb::add, dfb::recip_sum_exps>();
#else
        // compute 1/sum(exp(x))
        recip_tile_to_dfb<dfb::add, dfb::recip_sum_exps>();
#endif

        // step 3, compute final result
        dfb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_to_dfb<dfb::tmp, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            sub_tiles_to_dfb<dfb::max, dfb::in0, dfb::tmp>(0, 0, /*pop0=*/0, /*pop1=*/1);

            sub_tiles_to_dfb<dfb::tmp, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
