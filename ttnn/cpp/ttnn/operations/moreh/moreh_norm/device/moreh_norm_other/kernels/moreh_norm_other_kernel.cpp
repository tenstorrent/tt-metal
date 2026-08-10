// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
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

    constexpr uint32_t dfb_x_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_one_id = tt::CBIndex::c_1;
    DataflowBuffer dfb_one_obj(dfb_one_id);
    constexpr uint32_t dfb_decimal_id = tt::CBIndex::c_2;
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);
    constexpr uint32_t dfb_recip_p_decimal_id = tt::CBIndex::c_3;
    DataflowBuffer dfb_recip_p_decimal_obj(dfb_recip_p_decimal_id);

    constexpr uint32_t dfb_y_id = tt::CBIndex::c_16;

    constexpr uint32_t dfb_tmp0_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_tmp1_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_tmp2_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_tmp3_id = tt::CBIndex::c_27;
    constexpr uint32_t dfb_tmp4_id = tt::CBIndex::c_28;
    constexpr uint32_t dfb_tmp5_id = tt::CBIndex::c_29;

    constexpr uint32_t dfb_xabs_id = dfb_tmp0_id;
    constexpr uint32_t dfb_xpow_id = dfb_tmp1_id;
    constexpr uint32_t dfb_logx_id = dfb_tmp2_id;
    constexpr uint32_t dfb_exp_lxmd_id = dfb_tmp3_id;
    constexpr uint32_t dfb_correct_xpow_id = dfb_tmp4_id;
    constexpr uint32_t dfb_xpowadd_id = dfb_tmp5_id;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    dfb_one_obj.wait_front(onetile);
    dfb_decimal_obj.wait_front(onetile);
    dfb_recip_p_decimal_obj.wait_front(onetile);

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // |x|
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::CopyTile<
                    ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(
                    dfb_xabs_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            power_tile_to_dfb<
                dfb_xabs_id,
                dfb_xpow_id,
                dfb_logx_id,
                dfb_decimal_id,
                dfb_exp_lxmd_id,
                dfb_correct_xpow_id>(p, p_is_negative);

            if (inner_idx == 0) {
                copy_tile_to_dfb<dfb_correct_xpow_id, dfb_xpowadd_id>();
            } else {
                add_tiles_to_dfb<dfb_correct_xpow_id, dfb_xpowadd_id, dfb_xpowadd_id>();
            }
        }

        // Compute dfb_y_id
        power_tile_to_dfb<dfb_xpowadd_id, dfb_xabs_id, dfb_xpow_id, dfb_recip_p_decimal_id, dfb_logx_id, dfb_y_id>(
            recip_p, recip_p_is_negative);
    }
    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    dfb_recip_p_decimal_obj.pop_front(onetile);
}
