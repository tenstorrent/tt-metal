// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_1;
    constexpr uint32_t cb_decimal = tt::CBIndex::c_2;
    constexpr uint32_t cb_recip_p_decimal = tt::CBIndex::c_3;

    constexpr uint32_t cb_y = tt::CBIndex::c_16;

    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;

    constexpr uint32_t cb_xabs = cb_tmp0;
    constexpr uint32_t cb_xpow = cb_tmp1;
    constexpr uint32_t cb_logx = cb_tmp2;
    constexpr uint32_t cb_exp_lxmd = cb_tmp3;
    constexpr uint32_t cb_correct_xpow = cb_tmp4;
    constexpr uint32_t cb_xpowadd = cb_tmp5;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_decimal, onetile);
    cb_wait_front(cb_recip_p_decimal, onetile);

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // |x| — no mask in ord_other variant.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_xabs,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});

            // power_tile_to_cb 4 chain stages -> cb_correct_xpow.
            if (p_is_negative) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_xabs,
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
                        cb_xabs,
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
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_xabs,
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
                compute_kernel_lib::PackTile<
                    cb_correct_xpow,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});

            // Accumulator (cb_xpowadd): inner_idx==0 -> seed; else -> add.
            if (inner_idx == 0) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_correct_xpow,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_xpowadd,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_correct_xpow,
                        cb_xpowadd,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_xpowadd,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }
        }

        // Final |xpowadd|^(1/p) — power_tile_to_cb on cb_xpowadd -> cb_y.
        // Same 4-stage decomposition: cb_xpow=cb_tmp0, cb_logx=cb_tmp1,
        // cb_exp_lxmd=cb_tmp2, cb_correct_xpow=cb_y.
        if (recip_p_is_negative) {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_xpowadd,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{recip_p},
                compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_tmp0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        } else {
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_xpowadd,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PowerIterative<compute_kernel_lib::Dst::D0>{recip_p},
                compute_kernel_lib::PackTile<
                    cb_tmp0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_xpowadd,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::NoWaitPop,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_tmp1,
                cb_recip_p_decimal,
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
                cb_tmp2,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_tmp0,
                cb_tmp2,
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
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_recip_p_decimal, onetile);
}
