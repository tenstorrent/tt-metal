// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Abs, Sign
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
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

    constexpr uint32_t dfb_x_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_y_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_dy_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_decimal_id = tt::CBIndex::c_3;
    DataflowBuffer dfb_x_obj(dfb_x_id);
    DataflowBuffer dfb_y_obj(dfb_y_id);
    DataflowBuffer dfb_dy_obj(dfb_dy_id);
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);

    constexpr uint32_t dfb_dx_id = tt::CBIndex::c_16;

    constexpr uint32_t dfb_xpow_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_logx_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_exp_lxmd_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_correct_xpow_id = tt::CBIndex::c_27;
    constexpr uint32_t dfb_tmp4_id = tt::CBIndex::c_28;
    constexpr uint32_t dfb_tmp5_id = tt::CBIndex::c_29;
    constexpr uint32_t dfb_recip_ypow_id = tt::CBIndex::c_30;
    constexpr uint32_t dfb_sign_id = tt::CBIndex::c_31;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        dfb_x_obj.wait_front(onetile);   // comes from the reader
        dfb_y_obj.wait_front(onetile);   // comes from the reader
        dfb_dy_obj.wait_front(onetile);  // comes from the reader

        sign_tile_to_dfb<dfb_x_id, dfb_sign_id>(0, /*pop=*/0);

        // x^(p - 1)
        power_tile_with_abs_x_to_dfb<
            dfb_x_id,
            dfb_xpow_id,
            dfb_logx_id,
            dfb_decimal_id,
            dfb_exp_lxmd_id,
            dfb_correct_xpow_id>(p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> dfb_tmp4_id
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_correct_xpow_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb_y_id,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set)>{},
            ckl::PackTile<ckl::output(
                dfb_tmp4_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // x^(p - 1) * y * dy -> dfb_tmp5_id
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_tmp4_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb_dy_id,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set)>{},
            ckl::PackTile<ckl::output(
                dfb_tmp5_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // 1 / y^p
        power_and_recip_tile_to_dfb<
            dfb_y_id,
            dfb_xpow_id,
            dfb_logx_id,
            dfb_decimal_id,
            dfb_exp_lxmd_id,
            dfb_recip_ypow_id>(p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> dfb_dx_id
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_tmp5_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb_recip_ypow_id,
                    kBcast,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    kDataFormatReconfig)>{},
            ckl::PackTile<ckl::output(
                dfb_tmp4_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        dfb_dy_obj.pop_front(onetile);

        // multiply abs sign
        mul_tiles_to_dfb<dfb_sign_id, dfb_tmp4_id, dfb_dx_id>();
    }

    dfb_decimal_obj.pop_front(onetile);
}
