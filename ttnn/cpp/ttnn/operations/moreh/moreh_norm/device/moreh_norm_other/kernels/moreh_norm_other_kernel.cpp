// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

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
            ckl::unary<ckl::Abs<ckl::Dst::D0>, ckl::input(cb_x), ckl::output(cb_xabs)>(
                ckl::EltwiseShape::tiles(onetile));

            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_xabs, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{p},
                ckl::runtime_if(p_is_negative, ckl::Recip<ckl::Dst::D0>{}),
                ckl::PackTile<ckl::output(cb_xpow)>{});
            ckl::unary<
                ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
                ckl::input(cb_xabs, ckl::InputLifecycle::NoWaitPop),
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
            ckl::mul<ckl::input(cb_xpow), ckl::input(cb_exp_lxmd), ckl::output(cb_correct_xpow)>(
                ckl::EltwiseShape::tiles(onetile));

            if (inner_idx == 0) {
                ckl::copy<ckl::input(cb_correct_xpow), ckl::output(cb_xpowadd)>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<ckl::input(cb_correct_xpow), ckl::input(cb_xpowadd), ckl::output(cb_xpowadd)>(
                    ckl::EltwiseShape::tiles(onetile));
            }
        }

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<ckl::input(cb_xpowadd, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
            ckl::PowerIterative<ckl::Dst::D0>{recip_p},
            ckl::runtime_if(recip_p_is_negative, ckl::Recip<ckl::Dst::D0>{}),
            ckl::PackTile<ckl::output(cb_tmp0)>{});
        ckl::unary<
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
            ckl::input(cb_xpowadd, ckl::InputLifecycle::NoWaitPop),
            ckl::output(cb_tmp1)>(ckl::EltwiseShape::tiles(onetile));
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_tmp1),
                ckl::input(cb_recip_p_decimal, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp2)>{});
        ckl::mul<ckl::input(cb_tmp0), ckl::input(cb_tmp2), ckl::output(cb_y)>(ckl::EltwiseShape::tiles(onetile));
    }
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_recip_p_decimal, onetile);
}
