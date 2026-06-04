// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs, Sign
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    // Compile-time BroadcastDim from (ht_need_bcast, wt_need_bcast).
    constexpr auto kBcast = (ht_need_bcast && wt_need_bcast) ? compute_kernel_lib::BroadcastDim::Scalar
                            : ht_need_bcast                  ? compute_kernel_lib::BroadcastDim::Row
                            : wt_need_bcast                  ? compute_kernel_lib::BroadcastDim::Col
                                                             : compute_kernel_lib::BroadcastDim::None;

    // runtime args
    int i{0};
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto p_minus_one = get_arg_val<uint32_t>(i++);
    const bool p_minus_one_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;        // input(==x)
    constexpr uint32_t cb_y = tt::CBIndex::c_1;        // output(==y)
    constexpr uint32_t cb_dy = tt::CBIndex::c_2;       // output_grad(==dy)
    constexpr uint32_t cb_decimal = tt::CBIndex::c_3;  // decimal
    CircularBuffer cb_x_obj(cb_x);
    CircularBuffer cb_y_obj(cb_y);
    CircularBuffer cb_dy_obj(cb_dy);
    CircularBuffer cb_decimal_obj(cb_decimal);

    constexpr uint32_t cb_dx = tt::CBIndex::c_16;  // input_grad(==dx)

    constexpr uint32_t cb_xpow = tt::CBIndex::c_24;
    constexpr uint32_t cb_logx = tt::CBIndex::c_25;
    constexpr uint32_t cb_exp_lxmd = tt::CBIndex::c_26;
    constexpr uint32_t cb_correct_xpow = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;
    constexpr uint32_t cb_recip_ypow = tt::CBIndex::c_30;
    constexpr uint32_t cb_sign = tt::CBIndex::c_31;

    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        cb_x_obj.wait_front(onetile);
        cb_y_obj.wait_front(onetile);
        cb_dy_obj.wait_front(onetile);

        // sign(x) -> cb_sign. cb_x held (waited outside, no pop here).
        // Matches sign_tile_to_cb(cb_x, cb_sign, 0, pop=0) macro.
        compute_kernel_lib::unary<
            compute_kernel_lib::Sign<compute_kernel_lib::Dst::D0>,
            cb_x,
            cb_sign,
            compute_kernel_lib::CopyTileReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::HeldStream,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output>(onetile);

        // |x|^(p-1) — power_tile_with_abs_x_to_cb inlined as 4 chain stages.
        // Stage A: cb_x -> Abs -> Power(p-1) -> [Recip if neg] -> cb_xpow. InputLifecycle::HeldStream (no pop).
        // Stage B: cb_x -> Abs -> Log -> cb_logx. InputLifecycle::NoWaitPop (pops cb_x).
        // Stage C: cb_logx * cb_decimal -> Exp -> cb_exp_lxmd.
        // Stage D: cb_xpow * cb_exp_lxmd -> cb_correct_xpow.
        if (p_minus_one_is_negative) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p_minus_one},
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
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{p_minus_one},
                compute_kernel_lib::PackTile<
                    cb_xpow,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_x,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::NoWaitPop,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_logx,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
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
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exp_lxmd,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        compute_kernel_lib::mul<cb_xpow, cb_exp_lxmd, cb_correct_xpow>(onetile);

        // cb_correct_xpow * cb_y -> cb_tmp4. 4-branch bcast dispatch (compile-time).
        // cb_correct_xpow InputLifecycle::Streaming + Scalar (just pushed). cb_y InputLifecycle::CallerManaged (waited
        // outside).
        compute_kernel_lib::mul<
            cb_correct_xpow,
            cb_y,
            cb_tmp4,
            kBcast,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(onetile);

        // cb_tmp4 * cb_dy -> cb_tmp5. Same bcast pattern. cb_dy held outside loop.
        compute_kernel_lib::mul<
            cb_tmp4,
            cb_dy,
            cb_tmp5,
            kBcast,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(onetile);

        // 1 / y^p — power_and_recip_tile_to_cb inlined as 4 chain stages.
        // Stage A: cb_y -> Power(p) -> [Recip if neg] -> cb_xpow. InputLifecycle::HeldStream.
        // Stage B: cb_y -> Log -> cb_logx. InputLifecycle::NoWaitPop (pops cb_y).
        // Stage C: cb_logx * cb_decimal -> Exp -> cb_exp_lxmd.
        // Stage D: cb_xpow * cb_exp_lxmd -> Recip -> cb_recip_ypow.
        if (p_is_negative) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_y,
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
                    cb_y,
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
        compute_kernel_lib::unary<
            compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>,
            cb_y,
            cb_logx,
            compute_kernel_lib::CopyTileReconfig::Input,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::InputLifecycle::NoWaitPop,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output>(onetile);
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
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exp_lxmd,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
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
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_recip_ypow,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // (cb_tmp5 * cb_recip_ypow) -> cb_tmp4. Same 4-branch bcast.
        compute_kernel_lib::mul<cb_tmp5, cb_recip_ypow, cb_tmp4, kBcast>(onetile);

        cb_dy_obj.pop_front(onetile);

        // cb_sign * cb_tmp4 -> cb_dx. Final mul_tiles_to_cb inlined.
        compute_kernel_lib::mul<cb_sign, cb_tmp4, cb_dx>(onetile);
    }

    cb_decimal_obj.pop_front(onetile);
}
