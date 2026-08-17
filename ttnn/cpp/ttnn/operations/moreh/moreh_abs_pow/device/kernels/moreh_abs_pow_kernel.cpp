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

    constexpr uint32_t dfb_x_id = dfb::x;
    constexpr uint32_t dfb_one_id = dfb::one;
    constexpr uint32_t dfb_decimal_id = dfb::decimal;
    constexpr uint32_t dfb_mask_w_id = dfb::mask_w;
    constexpr uint32_t dfb_y_id = dfb::y;
    constexpr uint32_t dfb_xabs_id = dfb::xabs;          // |x|
    constexpr uint32_t dfb_xpow_id = dfb::xpow;          // |x|^p
    constexpr uint32_t dfb_logx_id = dfb::logx;          // log(|x|)
    constexpr uint32_t dfb_exp_lxmd_id = dfb::exp_lxmd;  // exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb_x_id, dfb_x_id, dfb_y_id);

    DataflowBuffer dfb_one_obj(dfb_one_id);
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);
    DataflowBuffer dfb_mask_w_obj(dfb_mask_w_id);

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
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<ckl::input(
                    dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    do_mask_w && (col_idx == Wt - 1),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{},
                    ckl::Mask<>{}),
                ckl::Abs<>{},
                ckl::PackTile<ckl::output(
                    dfb_xabs_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            power_tile_to_dfb<dfb_xabs_id, dfb_xpow_id, dfb_logx_id, dfb_decimal_id, dfb_exp_lxmd_id, dfb_y_id>(
                p, p_is_negative);
        }
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
