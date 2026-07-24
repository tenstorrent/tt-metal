// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Abs, Sign
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    // compile-time args
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
    DataflowBuffer cb_x_obj(cb_x);
    DataflowBuffer cb_y_obj(cb_y);
    DataflowBuffer cb_dy_obj(cb_dy);
    DataflowBuffer cb_decimal_obj(cb_decimal);

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

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        cb_x_obj.wait_front(onetile);   // comes from the reader
        cb_y_obj.wait_front(onetile);   // comes from the reader
        cb_dy_obj.wait_front(onetile);  // comes from the reader

        sign_tile_to_cb<cb_x, cb_sign>(0, /*pop=*/0);

        // x^(p - 1)
        power_tile_with_abs_x_to_cb<cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow>(
            p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> cb_tmp4
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_correct_xpow, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(
                    cb_y,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::BinaryFpuOp::Mul,
                kBcast>{},
            ckl::PackTile<ckl::output(cb_tmp4, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        // x^(p - 1) * y * dy -> cb_tmp5
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_tmp4, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(
                    cb_dy,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::BinaryFpuOp::Mul,
                kBcast>{},
            ckl::PackTile<ckl::output(cb_tmp5, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        // 1 / y^p
        power_and_recip_tile_to_cb<cb_y, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_recip_ypow>(p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> cb_dx
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_tmp5, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_recip_ypow, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::BinaryFpuOp::Mul,
                kBcast>{},
            ckl::PackTile<ckl::output(cb_tmp4, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        cb_dy_obj.pop_front(onetile);

        // multiply abs sign
        mul_tiles_to_cb<cb_sign, cb_tmp4, cb_dx>();
    }

    cb_decimal_obj.pop_front(onetile);
}
