// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Abs
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
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_1;
    DataflowBuffer dfb_one_obj(cb_one);
    constexpr uint32_t cb_decimal = tt::CBIndex::c_2;
    DataflowBuffer dfb_decimal_obj(cb_decimal);
    constexpr uint32_t cb_recip_p_decimal = tt::CBIndex::c_3;
    DataflowBuffer dfb_recip_p_decimal_obj(cb_recip_p_decimal);
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_4;
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);

    constexpr uint32_t cb_y = tt::CBIndex::c_16;

    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;
    constexpr uint32_t cb_tmp6 = tt::CBIndex::c_30;

    constexpr uint32_t cb_xabs = cb_tmp0;
    constexpr uint32_t cb_xpow = cb_tmp1;
    constexpr uint32_t cb_logx = cb_tmp2;
    constexpr uint32_t cb_exp_lxmd = cb_tmp3;
    constexpr uint32_t cb_correct_xpow = cb_tmp4;
    constexpr uint32_t cb_xpowadd = cb_tmp5;
    constexpr uint32_t cb_xpowsum = cb_tmp6;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    dfb_one_obj.wait_front(onetile);
    dfb_decimal_obj.wait_front(onetile);
    dfb_recip_p_decimal_obj.wait_front(onetile);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool mask_this = do_mask_w && (col_idx == Wt - 1);
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    mask_this,
                    ckl::CopyTile<
                        ckl::input(cb_mask_w, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{}),
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_xabs, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            power_tile_to_cb<cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow>(p, p_is_negative);

            if (col_idx == 0) {
                copy_tile_to_cb<cb_correct_xpow, cb_xpowadd>();
            } else {
                add_tiles_to_cb<cb_correct_xpow, cb_xpowadd, cb_xpowadd>();
            }
        }
        // Sum(|x|^p)
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_xpowadd, cb_one, cb_xpowsum>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        power_tile_to_cb<cb_xpowsum, cb_xabs, cb_xpow, cb_recip_p_decimal, cb_logx, cb_y>(recip_p, recip_p_is_negative);
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    dfb_recip_p_decimal_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
