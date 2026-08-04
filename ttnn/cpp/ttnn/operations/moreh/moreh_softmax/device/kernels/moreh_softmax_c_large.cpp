// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
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
    constexpr auto cb_in0 = dfb::in0;
    constexpr auto cb_out0 = dfb::out0;
    constexpr auto cb_exps = dfb::exps;
    constexpr auto cb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = dfb::add;
    constexpr auto cb_max = dfb::max;
    DataflowBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = dfb::tmp;

    constexpr uint32_t onetile = 1;

    uint32_t N = get_arg(args::N);
    uint32_t dim_size = get_arg(args::dim_size);

    compute_kernel_hw_startup(cb_in0, cb_exps, cb_out0);

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb<cb_in0, cb_max>();
            } else {
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    ckl::input(cb_in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(cb_max, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(cb_max, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::EltwiseShape::tiles(onetile));
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            sub_tiles_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb<cb_tmp, cb_exps>();
#else
            sub_tiles_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb<cb_tmp, cb_exps>();
#endif

            if (i == 0) {
                copy_tile_to_cb<cb_exps, cb_add>();
            } else {
                add_tiles_to_cb<cb_add, cb_exps, cb_add>();
            }
        }

#ifdef LOG
        // compute log(sum)
        log_tile_to_cb<cb_add, cb_recipsumexps>();
#else
        // compute 1/sum(exp(x))
        recip_tile_to_cb<cb_add, cb_recipsumexps>();
#endif

        cb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_to_cb<cb_tmp, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            sub_tiles_to_cb<cb_max, cb_in0, cb_tmp>(0, 0, /*pop0=*/0, /*pop1=*/1);

            sub_tiles_to_cb<cb_tmp, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb<cb_tmp, cb_exps>();

            mul_tiles_to_cb<cb_exps, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_to_cb<cb_in0, cb_max, cb_tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb<cb_tmp, cb_exps>();

            mul_tiles_to_cb<cb_exps, cb_recipsumexps, cb_out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
