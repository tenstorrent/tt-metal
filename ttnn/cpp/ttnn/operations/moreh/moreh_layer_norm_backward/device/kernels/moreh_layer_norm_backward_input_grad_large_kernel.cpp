// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add/sub/mul

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

void kernel_main() {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool gamma_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(5) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(6) == 1;

    compute_kernel_hw_startup(tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_16);

    constexpr auto dfb_dy_id = tt::CBIndex::c_0;
    constexpr auto dfb_x_id = tt::CBIndex::c_1;
    constexpr auto dfb_mean_id = tt::CBIndex::c_2;
    DataflowBuffer dfb_mean_obj(dfb_mean_id);  // mean
    constexpr auto dfb_rstd_id = tt::CBIndex::c_3;
    DataflowBuffer dfb_rstd_obj(dfb_rstd_id);  // rstd
    constexpr auto dfb_scaler_id = tt::CBIndex::c_4;
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);  // scaler
    constexpr auto dfb_n_recip_n_id = tt::CBIndex::c_5;
    DataflowBuffer dfb_n_recip_n_obj(dfb_n_recip_n_id);  // n_recip_n
    constexpr auto dfb_gamma_id = tt::CBIndex::c_6;
    constexpr auto dfb_mask_h_w_id = tt::CBIndex::c_7;
    DataflowBuffer dfb_mask_h_w_obj(dfb_mask_h_w_id);  // mask_h_w

    // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    constexpr auto dfb_dx_id = tt::CBIndex::c_16;

    // y = (x - mean) * rstd
    constexpr auto dfb_dycopy_id = tt::CBIndex::c_24;
    constexpr auto dfb_y_id = tt::CBIndex::c_25;
    constexpr auto dfb_dysum_id = tt::CBIndex::c_26;
    DataflowBuffer dfb_dysum_obj(dfb_dysum_id);  // Sum[dy]
    constexpr auto dfb_ydysum_id = tt::CBIndex::c_27;
    DataflowBuffer dfb_ydysum_obj(dfb_ydysum_id);  // Sum[y * dy]

    constexpr auto dfb_tmp1_id = tt::CBIndex::c_28;  // tmp1
    constexpr auto dfb_tmp2_id = tt::CBIndex::c_29;  // tmp2
    constexpr auto dfb_tmp3_id = tt::CBIndex::c_30;  // tmp3

    constexpr uint32_t onetile = 1;

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader
    dfb_n_recip_n_obj.wait_front(2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;
    constexpr auto gamma_bcast = is_groupnorm           ? ckl::BroadcastDim::Scalar
                                 : is_lastdim_layernorm ? ckl::BroadcastDim::Row
                                                        : ckl::BroadcastDim::None;

    if constexpr (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    constexpr uint32_t NCHt = num_rows_per_core;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        dfb_mean_obj.wait_front(onetile);  // comes from the reader
        dfb_rstd_obj.wait_front(onetile);  // comes from the reader

        // Compute dfb_y_id
        // y = (x - mean) * rstd
        constexpr auto dfb_dyadd_id = dfb_tmp1_id;
        constexpr auto dfb_ydyadd_id = dfb_tmp2_id;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute dfb_xmm_id
            // x - mean
            constexpr auto dfb_xmm_id = dfb_tmp3_id;
            DataflowBuffer dfb_xmm_obj(dfb_xmm_id);
            ckl::sub<
                ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_mean_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_y_id
            // (x - mean) * rstd and mask(optional)
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_rstd_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::BinaryFpuOp::Mul,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    dfb_y_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::OptionalChainElement<
                    gamma_has_value,
                    ckl::BinaryFpu<
                        ckl::input(dfb_dy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::input(
                            dfb_gamma_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::BinaryFpuOp::Mul,
                        gamma_bcast>>{},
                ckl::OptionalChainElement<
                    !gamma_has_value,
                    ckl::CopyTile<ckl::input(
                        dfb_dy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    dfb_dycopy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_dyadd_id
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb_dycopy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(
                        dfb_dyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            } else {
                ckl::add<
                    ckl::input(dfb_dyadd_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_dycopy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(
                        dfb_dyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            }
            // We don't pop dfb_dycopy_id here.

            // Compute dfb_ydy_id and dfb_ydyadd_id
            constexpr auto dfb_ydy_id = dfb_tmp3_id;
            ckl::mul<
                ckl::input(dfb_y_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_dycopy_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ydy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_ydyadd_id
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb_ydy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            } else {
                ckl::add<
                    ckl::input(dfb_ydyadd_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_ydy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            }
        }  // Wt loop

        // Compute dfb_dysum_id
        // Sum[dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_dyadd_id, dfb_scaler_id, dfb_dysum_id>(
            ckl::ReduceInputBlockShape::single());

        // Compute dfb_ydysum_id
        // Sum[y * dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_ydyadd_id, dfb_scaler_id, dfb_ydysum_id>(
            ckl::ReduceInputBlockShape::single());

        // Compute dfb_recip_nrstd_id
        // rstd / n -> dfb_tmp3_id
        constexpr auto dfb_recip_nrstd_id = dfb_tmp3_id;
        DataflowBuffer dfb_recip_nrstd_obj(dfb_recip_nrstd_id);
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(
                    dfb_n_recip_n_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::input(dfb_rstd_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::BinaryFpuOp::Mul,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{1u, 0u},
            ckl::PackTile<ckl::output(
                dfb_recip_nrstd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // Compute dfb_dx_id
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        dfb_dysum_obj.wait_front(onetile);
        dfb_ydysum_obj.wait_front(onetile);
        dfb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::OptionalChainElement<
                    gamma_has_value,
                    ckl::BinaryFpu<
                        ckl::input(dfb_dy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::input(
                            dfb_gamma_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::BinaryFpuOp::Mul,
                        gamma_bcast>>{},
                ckl::OptionalChainElement<
                    !gamma_has_value,
                    ckl::CopyTile<ckl::input(
                        dfb_dy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    dfb_dycopy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_ndy_id
            // n * dy
            constexpr auto dfb_ndy_id = dfb_tmp1_id;
            ckl::mul<
                ckl::input(dfb_n_recip_n_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::input(dfb_dycopy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ndy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_ndymdysum_id
            // n * dy - Sum[dy]
            constexpr auto dfb_ndymdysum_id = dfb_tmp2_id;
            ckl::sub<
                ckl::input(dfb_ndy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_dysum_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(
                    dfb_ndymdysum_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_xmm_id
            // x - mean and mask(optional)
            constexpr auto dfb_xmm_id = dfb_tmp1_id;
            DataflowBuffer dfb_xmm_obj(dfb_xmm_id);
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_mean_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::BinaryFpuOp::Sub,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_y_id
            ckl::mul<
                ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_rstd_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb_y_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_yydysum_id
            // y * Sum[y * dy]
            constexpr auto dfb_yydysum_id = dfb_tmp1_id;
            ckl::mul<
                ckl::input(dfb_y_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_ydysum_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb_yydysum_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_tmp4_id
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto dfb_tmp4_id = dfb_y_id;
            ckl::sub<
                ckl::input(dfb_ndymdysum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_yydysum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_tmp4_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute dfb_dx_id
            ckl::mul<
                ckl::input(dfb_tmp4_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_recip_nrstd_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb_dx_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
        }  // Wt loop
        dfb_recip_nrstd_obj.pop_front(onetile);
        dfb_dysum_obj.pop_front(onetile);
        dfb_ydysum_obj.pop_front(onetile);

        dfb_mean_obj.pop_front(onetile);
        dfb_rstd_obj.pop_front(onetile);
    }  // NCHt loop
    dfb_scaler_obj.pop_front(onetile);
    dfb_n_recip_n_obj.pop_front(2);

    if constexpr (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.pop_front(2);
    }
}
