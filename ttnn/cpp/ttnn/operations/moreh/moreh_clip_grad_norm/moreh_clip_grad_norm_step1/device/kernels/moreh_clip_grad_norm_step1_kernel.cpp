// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Abs
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_one = 1;
    constexpr uint32_t cb_decimal = 2;
    constexpr uint32_t cb_mask_h_w = 3;

    constexpr uint32_t cb_y = 16;

    constexpr uint32_t cb_xabs = 24;
    constexpr uint32_t cb_xpow = 25;
    constexpr uint32_t cb_xpowadd = 26;
    constexpr uint32_t cb_logx = 27;
    constexpr uint32_t cb_exp_lxmd = 28;
    constexpr uint32_t cb_correct_xpow = 29;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    const auto ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto wt = (origin_w + TILE_W - 1) / TILE_W;

    binary_op_init_common(cb_logx, cb_decimal, cb_y);

    cb_wait_front(cb_decimal, onetile);
    cb_wait_front(cb_one, onetile);

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);
    }

    // Compute cb_xpowadd
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const bool mh = do_mask_h && need_to_do_mask_h(tile_idx, ht, wt);
        const bool mw = do_mask_w && ((tile_idx + 1) % wt) == 0;
        if (mh && mw) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_x>{},
                ckl::CopyTile<cb_mask_h_w, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::CopyTile<
                    cb_mask_h_w,
                    ckl::Dst::D1,
                    ckl::input(
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Set)>{1u},  // mask_w lives at index 1 of cb_mask_h_w
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<
                    cb_xabs,
                    ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
        } else if (mh) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_x>{},
                ckl::CopyTile<cb_mask_h_w, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<
                    cb_xabs,
                    ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
        } else if (mw) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_x>{},
                ckl::CopyTile<
                    cb_mask_h_w,
                    ckl::Dst::D1,
                    ckl::input(
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Set)>{1u},  // mask_w lives at index 1 of cb_mask_h_w
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PackTile<
                    cb_xabs,
                    ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
        } else {
            ckl::unary<
                ckl::Abs<ckl::Dst::D0>,
                cb_x,
                cb_xabs,
                ckl::input(),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        }

        // |x + decimal|^p
        if (p_is_negative) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_xabs, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                ckl::PowerIterative<ckl::Dst::D0>{p},
                ckl::Recip<ckl::Dst::D0>{},
                ckl::PackTile<cb_xpow>{});
        } else {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_xabs, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                ckl::PowerIterative<ckl::Dst::D0>{p},
                ckl::PackTile<cb_xpow>{});
        }
        ckl::unary<
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
            cb_xabs,
            cb_logx,
            ckl::input(ckl::InputLifecycle::NoWaitPop)>(ckl::EltwiseShape::tiles(onetile));
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_logx,
                cb_decimal,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<cb_exp_lxmd>{});
        ckl::mul<cb_xpow, cb_exp_lxmd, cb_correct_xpow>(ckl::EltwiseShape::tiles(onetile));

        if (tile_idx == 0) {
            ckl::copy<
                cb_correct_xpow,
                cb_xpowadd,
                ckl::input(),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        } else {
            ckl::add<
                cb_correct_xpow,
                cb_xpowadd,
                cb_xpowadd,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        }
    }

    // Compute cb_y - reduce single pre-accumulated tile to scalar
    ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xpowadd, cb_one, cb_y>(ckl::ReduceInputBlockShape::single());

    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_one, onetile);
    if (do_mask_h || do_mask_w) {
        cb_pop_front(cb_mask_h_w, 2);
    }
}
