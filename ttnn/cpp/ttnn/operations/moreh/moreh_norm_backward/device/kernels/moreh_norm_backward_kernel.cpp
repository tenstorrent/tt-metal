// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Abs, Sign
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    // compile-time args
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    constexpr auto kBcast = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                            : ht_need_bcast                  ? ckl::BroadcastDim::Row
                            : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                             : ckl::BroadcastDim::None;

    // runtime args
    const auto num_input_tiles_per_core = get_arg(args::num_input_tiles_per_core);
    const auto p = get_arg(args::p);
    const bool p_is_negative = get_arg(args::p_is_negative) == 1;
    const auto p_minus_one = get_arg(args::p_minus_one);
    const bool p_minus_one_is_negative = get_arg(args::p_minus_one_is_negative) == 1;

    DataflowBuffer dfb_x_obj(dfb::x);              // input(==x), c_0
    DataflowBuffer dfb_y_obj(dfb::y);              // output(==y), c_1
    DataflowBuffer dfb_dy_obj(dfb::dy);            // output_grad(==dy), c_2
    DataflowBuffer dfb_decimal_obj(dfb::decimal);  // decimal, c_3

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::dx);
    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        dfb_x_obj.wait_front(onetile);   // comes from the reader
        dfb_y_obj.wait_front(onetile);   // comes from the reader
        dfb_dy_obj.wait_front(onetile);  // comes from the reader

        sign_tile_to_dfb<dfb::x, dfb::sign>(0, /*pop=*/0);

        // x^(p - 1)
        power_tile_with_abs_x_to_dfb<dfb::x, dfb::xpow, dfb::logx, dfb::decimal, dfb::exp_lxmd, dfb::correct_xpow>(
            p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> dfb::tmp4
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::correct_xpow, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::y,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp4, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // x^(p - 1) * y * dy -> dfb::tmp5
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::tmp4, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::dy,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp5, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // 1 / y^p
        power_and_recip_tile_to_dfb<dfb::y, dfb::xpow, dfb::logx, dfb::decimal, dfb::exp_lxmd, dfb::recip_ypow>(
            p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> dfb::tmp4
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::tmp5, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::recip_ypow, kBcast, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp4, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        dfb_dy_obj.pop_front(onetile);

        // multiply abs sign
        mul_tiles_to_dfb<dfb::sign, dfb::tmp4, dfb::dx>();
    }

    dfb_decimal_obj.pop_front(onetile);
}
