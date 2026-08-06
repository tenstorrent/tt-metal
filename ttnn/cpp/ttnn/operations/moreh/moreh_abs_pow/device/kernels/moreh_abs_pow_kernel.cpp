// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint.h"
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
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto origin_w = get_arg(args::origin_w);
    const auto p = get_arg(args::p);
    const bool p_is_negative = get_arg(args::p_is_negative) == 1;

    constexpr uint32_t cb_x = dfb::x;
    constexpr uint32_t cb_one = dfb::one;
    constexpr uint32_t cb_decimal = dfb::decimal;
    constexpr uint32_t cb_mask_w = dfb::mask_w;
    constexpr uint32_t cb_y = dfb::y;
    constexpr uint32_t cb_xabs = dfb::xabs;          // |x|
    constexpr uint32_t cb_xpow = dfb::xpow;          // |x|^p
    constexpr uint32_t cb_logx = dfb::logx;          // log(|x|)
    constexpr uint32_t cb_exp_lxmd = dfb::exp_lxmd;  // exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_x, cb_x, cb_y);

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
                ckl::CopyTile<ckl::input(
                    cb_x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    do_mask_w && (col_idx == Wt - 1),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_w,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{},
                    ckl::Mask<>{}),
                ckl::Abs<>{},
                ckl::PackTile<ckl::output(
                    cb_xabs, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            power_tile_to_cb<cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y>(p, p_is_negative);
        }
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
