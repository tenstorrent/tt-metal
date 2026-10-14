// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

ALWI bool need_to_do_mask_h(uint32_t row_idx, uint32_t Ht) { return (row_idx + 1) % Ht == 0; }

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t dfb_x_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_one_id = tt::CBIndex::c_1;
    DataflowBuffer dfb_one_obj(dfb_one_id);
    constexpr uint32_t dfb_decimal_id = tt::CBIndex::c_2;
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);
    constexpr uint32_t dfb_recip_p_decimal_id = tt::CBIndex::c_3;  // recip_p_decimal
    DataflowBuffer dfb_recip_p_decimal_obj(dfb_recip_p_decimal_id);
    constexpr uint32_t dfb_mask_h_id = tt::CBIndex::c_4;
    DataflowBuffer dfb_mask_h_obj(dfb_mask_h_id);

    constexpr uint32_t dfb_y_id = tt::CBIndex::c_16;

    constexpr uint32_t dfb_tmp0_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_tmp1_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_tmp2_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_tmp3_id = tt::CBIndex::c_27;
    constexpr uint32_t dfb_tmp4_id = tt::CBIndex::c_28;
    constexpr uint32_t dfb_tmp5_id = tt::CBIndex::c_29;
    constexpr uint32_t dfb_tmp6_id = tt::CBIndex::c_30;

    constexpr uint32_t dfb_xabs_id = dfb_tmp0_id;          // |x|
    constexpr uint32_t dfb_xpow_id = dfb_tmp1_id;          // |x|^p
    constexpr uint32_t dfb_logx_id = dfb_tmp2_id;          // log(|x|)
    constexpr uint32_t dfb_exp_lxmd_id = dfb_tmp3_id;      // exp(log(|x|) * decimal)
    constexpr uint32_t dfb_correct_xpow_id = dfb_tmp4_id;  // |x|^p * exp(log(|x|) * decimal)(==|x + decimal|^p)
    constexpr uint32_t dfb_xpowadd_id = dfb_tmp5_id;       // Add(|x + decimal|^p)
    constexpr uint32_t dfb_xpowsum_id = dfb_tmp6_id;       // Sum(|x + decimal|^p)

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    dfb_one_obj.wait_front(onetile);              // comes from the reader
    dfb_decimal_obj.wait_front(onetile);          // comes from the reader
    dfb_recip_p_decimal_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);  // comes from the reader
    }

    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            const bool mask_this = do_mask_h && need_to_do_mask_h(row_idx, Ht);
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(onetile),
                ckl::CopyTile<ckl::input(
                    dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    mask_this,
                    ckl::CopyTile<
                        ckl::input(dfb_mask_h_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{}),
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

            // Add(|x|^p)
            if (row_idx == 0) {
                copy_tile_to_dfb<dfb_correct_xpow_id, dfb_xpowadd_id>();
            } else {
                add_tiles_to_dfb<dfb_correct_xpow_id, dfb_xpowadd_id, dfb_xpowadd_id>();
            }
        }
        // Sum(|x|^p) - reduce single pre-accumulated tile
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb_xpowadd_id, dfb_one_id, dfb_xpowsum_id>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        power_tile_to_dfb<dfb_xpowsum_id, dfb_xabs_id, dfb_xpow_id, dfb_recip_p_decimal_id, dfb_logx_id, dfb_y_id>(
            recip_p, recip_p_is_negative);
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    dfb_recip_p_decimal_obj.pop_front(onetile);
    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
}
