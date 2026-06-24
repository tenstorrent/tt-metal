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

    constexpr uint32_t cb_input = 0;    // input(==tmp_pow_sum)
    constexpr uint32_t cb_decimal = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;  // output(==total_norm)

    constexpr uint32_t cb_x = 24;         // Sum[tmp_pow_sum](==x)
    constexpr uint32_t cb_xpow = 25;      // x^p
    constexpr uint32_t cb_logx = 26;      // log(x)
    constexpr uint32_t cb_exp_lxmd = 27;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x, cb_y);
    } else {
        binary_op_init_common(cb_logx, cb_decimal, cb_y);
    }

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            // Seed cb_x with first cb_input tile.
            // Original: copy_tile_init(cb_input) reconfigs srca; pack_tile no reconfig.
            ckl::copy<
                cb_input,
                cb_x,
                ckl::InputLifecycle::Streaming,
                ckl::OutputLifecycle::Streaming,
                ckl::CopyTileReconfig::Input,
                ckl::PackTileReconfig::None>(ckl::EltwiseShape::tiles(onetile));
        } else {
            // cb_x = cb_input + cb_x (in-place accumulator).
            // Original: add_tiles_init reconfigs srca/srcb; pack_tile no reconfig.
            ckl::add<
                cb_input,
                cb_x,
                cb_x,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::Streaming,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::None>(ckl::EltwiseShape::tiles(onetile));
        }
    }
    // Inline power_tile_to_cb body as 4 eltwise_chain stages:
    //   A: x^p  (InputLifecycle::HeldStream on cb_x → CopyTile + PowerIterative(p) + [Recip if p<0] +
    //   PackTile<cb_xpow>) B: log(x) (InputLifecycle::NoWaitPop on cb_x → CopyTile + Log + PackTile<cb_logx>) C:
    //   exp(log(x) * decimal) (BinaryFpu Mul + Exp + PackTile<cb_exp_lxmd>) D: xpow * exp_lxmd (BinaryFpu Mul +
    //   PackTile<cb_y>)
    //
    // Reconfig audit (matches power_tile_to_cb's per-stage *_with_dt calls):
    //   - copy_tile_init_with_dt -> CopyTileReconfig::Input
    //   - mul_tiles_init_with_dt -> BinaryDataFormatReconfig::Input
    //   - pack_tile_with_dt      -> PackTileReconfig::Output
    //
    // cb_decimal InputLifecycle::CallerManaged + Scalar (held by external wait_front at top of kernel).
    if (p_is_negative) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<cb_x, ckl::Dst::D0, ckl::InputLifecycle::HeldStream>{},
            ckl::PowerIterative<ckl::Dst::D0>{p},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_xpow>{});
    } else {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<cb_x, ckl::Dst::D0, ckl::InputLifecycle::HeldStream>{},
            ckl::PowerIterative<ckl::Dst::D0>{p},
            ckl::PackTile<cb_xpow>{});
    }

    // Stage B: log(x). cb_x already waited (Stage A's InputLifecycle::HeldStream did wait but no pop);
    //   now pop it via InputLifecycle::NoWaitPop.
    ckl::unary<ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>, cb_x, cb_logx, ckl::InputLifecycle::NoWaitPop>(
        ckl::EltwiseShape::tiles(onetile));

    // Stage C: exp(log(x) * decimal). cb_decimal pre-waited at top of kernel.
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(onetile),
        ckl::BinaryFpu<
            cb_logx,
            cb_decimal,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::CallerManaged>{},
        ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<cb_exp_lxmd>{});

    // Stage D: x^p * exp(log(x) * decimal) = (x + decimal)^p -> cb_y.
    ckl::mul<cb_xpow, cb_exp_lxmd, cb_y>(ckl::EltwiseShape::tiles(onetile));
}
