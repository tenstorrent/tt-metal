// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Abs
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI bool need_to_do_mask_h(uint32_t row_idx, uint32_t Ht) { return (row_idx + 1) % Ht == 0; }

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_1;
    constexpr uint32_t cb_decimal = tt::CBIndex::c_2;
    constexpr uint32_t cb_recip_p_decimal = tt::CBIndex::c_3;
    constexpr uint32_t cb_mask_h = tt::CBIndex::c_4;

    constexpr uint32_t cb_y = tt::CBIndex::c_16;

    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;
    constexpr uint32_t cb_tmp6 = tt::CBIndex::c_30;

    constexpr uint32_t cb_xabs = cb_tmp0;
    constexpr uint32_t cb_xpow = cb_tmp1;
    constexpr uint32_t cb_logx = cb_tmp2;
    constexpr uint32_t cb_exp_lxmd = cb_tmp3;
    constexpr uint32_t cb_correct_xpow = cb_tmp4;
    constexpr uint32_t cb_xpowadd = cb_tmp5;
    constexpr uint32_t cb_xpowsum = cb_tmp6;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_decimal, onetile);
    cb_wait_front(cb_recip_p_decimal, onetile);

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }

    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            if (do_mask_h && need_to_do_mask_h(row_idx, Ht)) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(cb_x)>{},
                    ckl::CopyTile<ckl::input(cb_mask_h, ckl::InputLifecycle::CallerManaged), ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::Abs<ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(cb_xabs)>{});
            } else {
                ckl::unary<ckl::Abs<ckl::Dst::D0>, ckl::input(cb_x), ckl::output(cb_xabs)>(
                    ckl::EltwiseShape::tiles(onetile));
            }

            if (p_is_negative) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(cb_xabs, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                    ckl::PowerIterative<ckl::Dst::D0>{p},
                    ckl::Recip<ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(cb_xpow)>{});
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(cb_xabs, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                    ckl::PowerIterative<ckl::Dst::D0>{p},
                    ckl::PackTile<ckl::output(cb_xpow)>{});
            }
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

            if (row_idx == 0) {
                ckl::copy<ckl::input(cb_correct_xpow), ckl::output(cb_xpowadd)>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<ckl::input(cb_correct_xpow), ckl::input(cb_xpowadd), ckl::output(cb_xpowadd)>(
                    ckl::EltwiseShape::tiles(onetile));
            }
        }

        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xpowadd, cb_one, cb_xpowsum>(ckl::ReduceInputBlockShape::single());

        if (recip_p_is_negative) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_xpowsum, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{recip_p},
                ckl::Recip<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_tmp0)>{});
        } else {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_xpowsum, ckl::InputLifecycle::HeldStream), ckl::Dst::D0>{},
                ckl::PowerIterative<ckl::Dst::D0>{recip_p},
                ckl::PackTile<ckl::output(cb_tmp0)>{});
        }
        ckl::unary<
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>,
            ckl::input(cb_xpowsum, ckl::InputLifecycle::NoWaitPop),
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
    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
}
