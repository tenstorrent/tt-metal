// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr uint32_t num_cols_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t NCHt = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_grad_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(7) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(8) == 1;

    constexpr auto dfb_dy_id = tt::CBIndex::c_0;
    constexpr auto dfb_x_id = tt::CBIndex::c_1;
    constexpr auto dfb_mean_id = tt::CBIndex::c_2;
    constexpr auto dfb_rstd_id = tt::CBIndex::c_3;
    constexpr auto dfb_scaler_id = tt::CBIndex::c_4;
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);  // scaler
    constexpr auto dfb_mask_h_id = tt::CBIndex::c_5;
    DataflowBuffer dfb_mask_h_obj(dfb_mask_h_id);  // mask_h
    constexpr auto dfb_mask_w_id = tt::CBIndex::c_6;
    DataflowBuffer dfb_mask_w_obj(dfb_mask_w_id);  // mask_w

    // Sum[y * dy]
    constexpr auto dfb_dgamma_id = tt::CBIndex::c_16;
    // Sum[dy]
    constexpr auto dfb_dbeta_id = tt::CBIndex::c_17;

    // y = (x - mean) * rstd
    constexpr auto dfb_y_id = tt::CBIndex::c_24;
    constexpr auto dfb_ydy_id = tt::CBIndex::c_25;
    constexpr auto dfb_dyadd_id = tt::CBIndex::c_26;
    constexpr auto dfb_ydyadd_id = tt::CBIndex::c_27;
    constexpr auto dfb_xmm_id = tt::CBIndex::c_28;
    constexpr auto dfb_dycopy_id = tt::CBIndex::c_29;
    DataflowBuffer dfb_dycopy_obj(dfb_dycopy_id);  // dycopy

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm);
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t Ht = origin_Ht;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr auto dfb_out_init_id = gamma_grad_has_value ? dfb_dgamma_id : dfb_dbeta_id;
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, dfb_out_init_id);

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }
    if constexpr (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    uint32_t h_idx;
    uint32_t w_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_cols_per_core; outer_idx++) {
        for (uint32_t inner_idx = 0; inner_idx < NCHt; inner_idx++) {
            if constexpr (is_groupnorm) {
                h_idx = (inner_idx % HtWt) / Wt;
                w_idx = (inner_idx % HtWt) % Wt;
            } else {
                h_idx = inner_idx;
                w_idx = outer_idx;
            }

            // Compute dfb_dycopy_id
            // deepcopy and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<ckl::input(
                    dfb_dy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    do_mask_h && ((h_idx + 1) % origin_Ht == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_h_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            dfb_mask_w_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    dfb_dycopy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_dyadd_id
            if constexpr (beta_grad_has_value) {
                if (inner_idx == 0) {
                    copy_tile_to_dfb<dfb_dycopy_id, dfb_dyadd_id>(0, gamma_grad_has_value ? 0 : 1);
                } else {
                    add_tiles_to_dfb<dfb_dyadd_id, dfb_dycopy_id, dfb_dyadd_id>(0, 0, 1, gamma_grad_has_value ? 0 : 1);
                }
            }  // beta_grad_has_value
            // We don't pop dfb_dycopy_id here.

            if constexpr (gamma_grad_has_value) {
                // Compute dfb_xmm_id
                // x - mean and mask(optional)
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Sub,
                        ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::input(
                            dfb_mean_id,
                            is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                            ckl::WaitPolicy::PerTile,
                            ckl::PopPolicy::PerTile,
                            kDataFormatReconfig)>{},
                    ckl::runtime_if(
                        do_mask_h && ((h_idx + 1) % origin_Ht == 0),
                        ckl::CopyTile<
                            ckl::input(
                                dfb_mask_h_id,
                                ckl::WaitPolicy::None,
                                ckl::PopPolicy::None,
                                ckl::OperandKind::Scalar,
                                kDataFormatReconfig,
                                ckl::TileOffset::Set),
                            ckl::Dst::D1>{0},
                        ckl::Mask<>{}),
                    ckl::runtime_if(
                        do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                        ckl::CopyTile<
                            ckl::input(
                                dfb_mask_w_id,
                                ckl::WaitPolicy::None,
                                ckl::PopPolicy::None,
                                ckl::OperandKind::Scalar,
                                kDataFormatReconfig,
                                ckl::TileOffset::Set),
                            ckl::Dst::D1>{0},
                        ckl::Mask<>{}),
                    ckl::PackTile<ckl::output(
                        dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

                // Compute dfb_y_id
                // (x - mean) * rstd
                ckl::mul<
                    ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(
                        dfb_rstd_id,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        kDataFormatReconfig),
                    ckl::output(dfb_y_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile());

                // Compute dfb_ydy_id
                mul_tiles_to_dfb<dfb_y_id, dfb_dycopy_id, dfb_ydy_id>(0, 0, 1, beta_grad_has_value ? 0 : 1);

                // Compute dfb_ydyadd_id
                if (inner_idx == 0) {
                    copy_tile_to_dfb<dfb_ydy_id, dfb_ydyadd_id>();
                } else {
                    add_tiles_to_dfb<dfb_ydyadd_id, dfb_ydy_id, dfb_ydyadd_id>();
                }
            }  // gamma_grad_has_value

            if constexpr (gamma_grad_has_value && beta_grad_has_value) {
                dfb_dycopy_obj.pop_front(onetile);
            }
        }  // inner_idx loop

        if constexpr (gamma_grad_has_value) {
            // Compute dfb_dgamma_id
            if constexpr (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb_ydyadd_id, dfb_scaler_id, dfb_dgamma_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                copy_tile_to_dfb<dfb_ydyadd_id, dfb_dgamma_id>();
            }
        }  // gamma_grad_has_value

        if constexpr (beta_grad_has_value) {
            // Compute dfb_dbeta_id
            if constexpr (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb_dyadd_id, dfb_scaler_id, dfb_dbeta_id>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                copy_tile_to_dfb<dfb_dyadd_id, dfb_dbeta_id>();
            }
        }  // beta_grad_has_value

    }  // outer_idx loop
    dfb_scaler_obj.pop_front(onetile);

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    if constexpr (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
