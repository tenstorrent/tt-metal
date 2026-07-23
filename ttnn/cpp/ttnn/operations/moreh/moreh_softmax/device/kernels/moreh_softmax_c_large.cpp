// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    DataflowBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    DataflowBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps, cb_out0);

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                copy_tile_to_cb<cb_in0, cb_max>();
            } else {
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    ckl::input(cb_in0, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::input(cb_max, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::output(cb_max, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
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
