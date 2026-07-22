// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Abs, Sign
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto kBcast = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                            : ht_need_bcast                  ? ckl::BroadcastDim::Row
                            : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                             : ckl::BroadcastDim::None;

    // runtime args
    int i{0};
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto p_minus_one = get_arg_val<uint32_t>(i++);
    const bool p_minus_one_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_y = tt::CBIndex::c_1;
    constexpr uint32_t cb_dy = tt::CBIndex::c_2;
    constexpr uint32_t cb_decimal = tt::CBIndex::c_3;
    CircularBuffer cb_x_obj(cb_x);
    CircularBuffer cb_y_obj(cb_y);
    CircularBuffer cb_dy_obj(cb_dy);
    CircularBuffer cb_decimal_obj(cb_decimal);

    constexpr uint32_t cb_dx = tt::CBIndex::c_16;

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

        ckl::unary<ckl::Sign<ckl::Dst::D0>, ckl::input(cb_x, ckl::InputLifecycle::HeldStream), ckl::output(cb_sign)>(
            ckl::EltwiseShape::tiles(onetile));

        if (p_minus_one_is_negative) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{p_minus_one},
                ckl::Recip<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_xpow)>{});
        } else {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::Abs<ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{p_minus_one},
                ckl::PackTile<ckl::output(cb_xpow)>{});
        }
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<ckl::input(cb_x, ckl::InputLifecycle::NoWaitPop), ckl::Dst::D0>{},
            ckl::Abs<ckl::Dst::D0>{},
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_logx)>{});
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_logx),
                ckl::input(cb_decimal, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_exp_lxmd)>{});
        ckl::mul<ckl::input(cb_xpow), ckl::input(cb_exp_lxmd), ckl::output(cb_correct_xpow)>(
            ckl::EltwiseShape::tiles(onetile));

        ckl::mul<
            ckl::input(cb_correct_xpow),
            ckl::input(cb_y, ckl::InputLifecycle::CallerManaged),
            ckl::output(cb_tmp4),
            kBcast>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<
            ckl::input(cb_tmp4),
            ckl::input(cb_dy, ckl::InputLifecycle::CallerManaged),
            ckl::output(cb_tmp5),
            kBcast>(ckl::EltwiseShape::tiles(onetile));

        if (p_is_negative) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_y, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{p},
                ckl::Recip<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_xpow)>{});
        } else {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_y, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{p},
                ckl::PackTile<ckl::output(cb_xpow)>{});
        }
        ckl::unary<
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
            ckl::input(cb_y, ckl::InputLifecycle::NoWaitPop),
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
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<ckl::input(cb_xpow), ckl::input(cb_exp_lxmd), ckl::BinaryFpuOp::Mul>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_ypow)>{});

        ckl::mul<ckl::input(cb_tmp5), ckl::input(cb_recip_ypow), ckl::output(cb_tmp4), kBcast>(
            ckl::EltwiseShape::tiles(onetile));

        cb_dy_obj.pop_front(onetile);

        ckl::mul<ckl::input(cb_sign), ckl::input(cb_tmp4), ckl::output(cb_dx)>(ckl::EltwiseShape::tiles(onetile));
    }

    cb_decimal_obj.pop_front(onetile);
}
