// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
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
                ckl::input(cb_input),
                ckl::output(cb_x, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        } else {
            ckl::add<
                ckl::input(cb_input),
                ckl::input(cb_x),
                ckl::output(cb_x, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
        }
    }
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(onetile),
        ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
        ckl::PowerIterative<ckl::Dst::D0>{p},
        ckl::runtime_if(p_is_negative, ckl::Recip<ckl::Dst::D0>{}),
        ckl::PackTile<ckl::output(cb_xpow)>{});

    ckl::unary<
        ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
        ckl::input(cb_x, ckl::InputLifecycle::NoWaitPop),
        ckl::output(cb_logx)>(ckl::EltwiseShape::tiles(onetile));

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(onetile),
        ckl::BinaryFpu<
            ckl::input(cb_logx),
            ckl::input(cb_decimal, ckl::InputLifecycle::CallerManaged),
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::None>{},
        ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_exp_lxmd)>{});

    ckl::mul<ckl::input(cb_xpow), ckl::input(cb_exp_lxmd), ckl::output(cb_y)>(ckl::EltwiseShape::tiles(onetile));
}
