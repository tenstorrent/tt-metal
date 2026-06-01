// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

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
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_input,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else {
            // cb_x = cb_input + cb_x (in-place accumulator).
            // Original: add_tiles_init reconfigs srca/srcb; pack_tile no reconfig.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_input,
                    cb_x,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
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
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_x,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p},
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_xpow,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    } else {
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_x,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p},
            compute_kernel_lib::PackTile<
                cb_xpow,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }

    // Stage B: log(x). cb_x already waited (Stage A's InputLifecycle::HeldStream did wait but no pop);
    //   now pop it via InputLifecycle::NoWaitPop.
    compute_kernel_lib::eltwise_chain(
        onetile,
        compute_kernel_lib::CopyTile<
            cb_x,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::NoWaitPop,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::Input>{},
        compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_logx,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output>{});

    // Stage C: exp(log(x) * decimal). cb_decimal pre-waited at top of kernel.
    compute_kernel_lib::eltwise_chain(
        onetile,
        compute_kernel_lib::BinaryFpu<
            cb_logx,
            cb_decimal,
            compute_kernel_lib::BinaryFpuOp::Mul,
            compute_kernel_lib::BroadcastDim::None,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::
            Exp<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_exp_lxmd,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output>{});

    // Stage D: x^p * exp(log(x) * decimal) = (x + decimal)^p -> cb_y.
    compute_kernel_lib::eltwise_chain(
        onetile,
        compute_kernel_lib::BinaryFpu<
            cb_xpow,
            cb_exp_lxmd,
            compute_kernel_lib::BinaryFpuOp::Mul,
            compute_kernel_lib::BroadcastDim::None,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::PackTile<
            cb_y,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output>{});
}
