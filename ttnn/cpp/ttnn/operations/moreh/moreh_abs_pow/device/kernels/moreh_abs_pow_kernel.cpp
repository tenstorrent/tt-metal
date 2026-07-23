// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint.h"
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

    constexpr auto cb_x = tt::CB::c_in0;
    constexpr auto cb_one = tt::CB::c_in1;
    constexpr auto cb_decimal = tt::CB::c_in2;
    constexpr auto cb_mask_w = tt::CB::c_in3;

    constexpr auto cb_y = tt::CB::c_out0;

    constexpr auto cb_tmp0 = tt::CB::c_intermed0;
    constexpr auto cb_tmp1 = tt::CB::c_intermed1;
    constexpr auto cb_tmp2 = tt::CB::c_intermed2;
    constexpr auto cb_tmp3 = tt::CB::c_intermed3;

    constexpr auto cb_xabs = cb_tmp0;      // |x|
    constexpr auto cb_xpow = cb_tmp1;      // |x|^p
    constexpr auto cb_logx = cb_tmp2;      // log(|x|)
    constexpr auto cb_exp_lxmd = cb_tmp3;  // exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    DataflowBuffer dfb_one_obj(cb_one);
    DataflowBuffer dfb_decimal_obj(cb_decimal);
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);

    dfb_one_obj.wait_front(onetile);
    dfb_decimal_obj.wait_front(onetile);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }
    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    do_mask_w && (col_idx == Wt - 1),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{},
                    ckl::Mask<>{}),
                ckl::Abs<>{},
                ckl::PackTile<ckl::output(cb_xabs, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            power_tile_to_cb<cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y>(p, p_is_negative);
        }
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
