// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_1;
    constexpr uint32_t cb_decimal = tt::CBIndex::c_2;
    constexpr uint32_t cb_recip_p_decimal = tt::CBIndex::c_3;

    constexpr uint32_t cb_y = tt::CBIndex::c_16;

    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;

    constexpr uint32_t cb_xabs = cb_tmp0;
    constexpr uint32_t cb_xpow = cb_tmp1;
    constexpr uint32_t cb_logx = cb_tmp2;
    constexpr uint32_t cb_exp_lxmd = cb_tmp3;
    constexpr uint32_t cb_correct_xpow = cb_tmp4;
    constexpr uint32_t cb_xpowadd = cb_tmp5;

    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_decimal, onetile);
    cb_wait_front(cb_recip_p_decimal, onetile);

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // |x|
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::Streaming, kDataFormatReconfig), ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_xabs, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            power_tile_to_cb<cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow>(p, p_is_negative);

            if (inner_idx == 0) {
                copy_tile_to_cb<cb_correct_xpow, cb_xpowadd>();
            } else {
                add_tiles_to_cb<cb_correct_xpow, cb_xpowadd, cb_xpowadd>();
            }
        }

        // Compute cb_y
        power_tile_to_cb<cb_xpowadd, cb_xabs, cb_xpow, cb_recip_p_decimal, cb_logx, cb_y>(recip_p, recip_p_is_negative);
    }
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_recip_p_decimal, onetile);
}
