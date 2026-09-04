// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t dfb_x_id = 0;
    constexpr uint32_t dfb_one_id = 1;
    DataflowBuffer dfb_one_obj(dfb_one_id);
    constexpr uint32_t dfb_decimal_id = 2;
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);
    constexpr uint32_t dfb_mask_h_w_id = 3;
    DataflowBuffer dfb_mask_h_w_obj(dfb_mask_h_w_id);

    constexpr uint32_t dfb_y_id = 16;

    constexpr uint32_t dfb_xabs_id = 24;          // |x|
    constexpr uint32_t dfb_xpow_id = 25;          // |x|^p
    constexpr uint32_t dfb_xpowadd_id = 26;       // Add[|x|^p * exp(log(|x|) * decimal)]
    constexpr uint32_t dfb_logx_id = 27;          // log(|x|)
    constexpr uint32_t dfb_exp_lxmd_id = 28;      // exp(log(|x|) * decimal)
    constexpr uint32_t dfb_correct_xpow_id = 29;  // |x|^p * exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t mask_w_tile_index = 1;

#if defined FP32_DEST_ACC_EN
    constexpr auto data_format_reconfig = ckl::DataFormatReconfig::Enabled;
#else
    constexpr auto data_format_reconfig = ckl::DataFormatReconfig::Disabled;
#endif

    using CopyMaskH = ckl::CopyTile<
        ckl::input(dfb_mask_h_w_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, data_format_reconfig),
        ckl::Dst::D1>;
    using CopyMaskW = ckl::CopyTile<
        ckl::input(
            dfb_mask_h_w_id,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Scalar,
            data_format_reconfig,
            ckl::TileAddressing::Offset),
        ckl::Dst::D1>;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    const auto ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto wt = (origin_w + TILE_W - 1) / TILE_W;

    compute_kernel_hw_startup(dfb_logx_id, dfb_decimal_id, dfb_y_id);

    dfb_decimal_obj.wait_front(onetile);  // comes from the reader
    dfb_one_obj.wait_front(onetile);      // comes from the reader

    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    // Compute dfb_xpowadd_id
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Compute dfb_xabs_id and mask(optional)
        const bool mh = do_mask_h && need_to_do_mask_h(tile_idx, ht, wt);
        const bool mw = do_mask_w && ((tile_idx + 1) % wt) == 0;
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<ckl::input(
                dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, data_format_reconfig)>{},
            ckl::runtime_if(mh, CopyMaskH{}, ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{}),
            ckl::runtime_if(mw, CopyMaskW{mask_w_tile_index}, ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{}),
            ckl::Abs<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_xabs_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, data_format_reconfig)>{});

        // |x + decimal|^p
        power_tile_to_dfb<dfb_xabs_id, dfb_xpow_id, dfb_logx_id, dfb_decimal_id, dfb_exp_lxmd_id, dfb_correct_xpow_id>(
            p, p_is_negative);

        if (tile_idx == 0) {
            ckl::copy<
                ckl::input(
                    dfb_correct_xpow_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, data_format_reconfig),
                ckl::output(
                    dfb_xpowadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, data_format_reconfig)>(
                ckl::IterationShape::one_tile());
        } else {
            ckl::add<
                ckl::input(
                    dfb_correct_xpow_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, data_format_reconfig),
                ckl::input(dfb_xpowadd_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, data_format_reconfig),
                ckl::output(
                    dfb_xpowadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, data_format_reconfig)>(
                ckl::IterationShape::one_tile());
        }
    }

    // Compute dfb_y_id - reduce single pre-accumulated tile to scalar
    ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_xpowadd_id, dfb_one_id, dfb_y_id>(ckl::ReduceInputBlockShape::single());

    dfb_decimal_obj.pop_front(onetile);
    dfb_one_obj.pop_front(onetile);
    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.pop_front(2);
    }
}
