// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_decimal = 1;

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;

    constexpr uint32_t cb_x = 24;
    constexpr uint32_t cb_xpow = 25;
    constexpr uint32_t cb_logx = 26;
    constexpr uint32_t cb_exp_lxmd = 27;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x, cb_y);
    } else {
        binary_op_init_common(cb_logx, cb_decimal, cb_y);
    }

    cb_wait_front(cb_decimal, onetile);

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            ckl::copy<
                cb_input,
                cb_x,
                ckl::input(),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        } else {
            ckl::add<
                cb_input,
                cb_x,
                cb_x,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        }
    }
    if (p_is_negative) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<cb_x, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::HeldStream)>{},
            ckl::PowerIterative<ckl::Dst::D0>{p},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_xpow>{});
    } else {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<cb_x, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::HeldStream)>{},
            ckl::PowerIterative<ckl::Dst::D0>{p},
            ckl::PackTile<cb_xpow>{});
    }

    ckl::unary<ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>, cb_x, cb_logx, ckl::input(ckl::InputLifecycle::NoWaitPop)>(
        ckl::EltwiseShape::tiles(onetile));

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

    ckl::mul<cb_xpow, cb_exp_lxmd, cb_y>(ckl::EltwiseShape::tiles(onetile));
}
