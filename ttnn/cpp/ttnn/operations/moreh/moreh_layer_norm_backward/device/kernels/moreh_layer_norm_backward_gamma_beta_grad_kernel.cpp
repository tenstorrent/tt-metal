// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

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

    constexpr auto cb_dy = tt::CBIndex::c_0;
    constexpr auto cb_x = tt::CBIndex::c_1;
    constexpr auto cb_mean = tt::CBIndex::c_2;
    constexpr auto cb_rstd = tt::CBIndex::c_3;
    constexpr auto cb_scaler = tt::CBIndex::c_4;
    DataflowBuffer dfb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    DataflowBuffer dfb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);  // mask_w

    // Sum[y * dy]
    constexpr auto cb_dgamma = tt::CBIndex::c_16;
    // Sum[dy]
    constexpr auto cb_dbeta = tt::CBIndex::c_17;

    // y = (x - mean) * rstd
    constexpr auto cb_y = tt::CBIndex::c_24;
    constexpr auto cb_ydy = tt::CBIndex::c_25;
    constexpr auto cb_dyadd = tt::CBIndex::c_26;
    constexpr auto cb_ydyadd = tt::CBIndex::c_27;
    constexpr auto cb_xmm = tt::CBIndex::c_28;
    constexpr auto cb_dycopy = tt::CBIndex::c_29;
    DataflowBuffer dfb_dycopy_obj(cb_dycopy);  // dycopy

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm);
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t Ht = origin_Ht;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, cb_out_init);

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

            // Compute cb_dycopy
            // deepcopy and mask(optional)
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::CopyTile<ckl::input(cb_dy, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                ckl::runtime_if(
                    do_mask_h && ((h_idx + 1) % origin_Ht == 0),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_h,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(cb_dycopy, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            // Compute cb_dyadd
            if constexpr (beta_grad_has_value) {
                if (inner_idx == 0) {
                    copy_tile_to_cb<cb_dycopy, cb_dyadd>(0, gamma_grad_has_value ? 0 : 1);
                } else {
                    add_tiles_to_cb<cb_dyadd, cb_dycopy, cb_dyadd>(0, 0, 1, gamma_grad_has_value ? 0 : 1);
                }
            }  // beta_grad_has_value
            // We don't pop cb_dycopy here.

            if constexpr (gamma_grad_has_value) {
                // Compute cb_xmm
                // x - mean and mask(optional)
                ckl::eltwise_chain(
                    ckl::EltwiseShape::single(),
                    ckl::BinaryFpu<
                        ckl::input(cb_x, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::input(cb_mean, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::BinaryFpuOp::Sub,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{},
                    ckl::runtime_if(
                        do_mask_h && ((h_idx + 1) % origin_Ht == 0),
                        ckl::CopyTile<
                            ckl::input(
                                cb_mask_h,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::OperandKind::Scalar,
                                kDataFormatReconfig,
                                ckl::TileOffset::Set),
                            ckl::Dst::D1>{0},
                        ckl::Mask<>{}),
                    ckl::runtime_if(
                        do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                        ckl::CopyTile<
                            ckl::input(
                                cb_mask_w,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::OperandKind::Scalar,
                                kDataFormatReconfig,
                                ckl::TileOffset::Set),
                            ckl::Dst::D1>{0},
                        ckl::Mask<>{}),
                    ckl::PackTile<ckl::output(cb_xmm, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

                // Compute cb_y
                // (x - mean) * rstd
                ckl::mul<
                    ckl::input(cb_xmm, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::input(cb_rstd, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::output(cb_y, ckl::OutputLifecycle::Streaming, kDataFormatReconfig),
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                    ckl::EltwiseShape::single());

                // Compute cb_ydy
                mul_tiles_to_cb<cb_y, cb_dycopy, cb_ydy>(0, 0, 1, beta_grad_has_value ? 0 : 1);

                // Compute cb_ydyadd
                if (inner_idx == 0) {
                    copy_tile_to_cb<cb_ydy, cb_ydyadd>();
                } else {
                    add_tiles_to_cb<cb_ydyadd, cb_ydy, cb_ydyadd>();
                }
            }  // gamma_grad_has_value

            if constexpr (gamma_grad_has_value && beta_grad_has_value) {
                dfb_dycopy_obj.pop_front(onetile);
            }
        }  // inner_idx loop

        if constexpr (gamma_grad_has_value) {
            // Compute cb_dgamma
            if constexpr (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_ydyadd, cb_scaler, cb_dgamma>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                copy_tile_to_cb<cb_ydyadd, cb_dgamma>();
            }
        }  // gamma_grad_has_value

        if constexpr (beta_grad_has_value) {
            // Compute cb_dbeta
            if constexpr (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_dyadd, cb_scaler, cb_dbeta>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                copy_tile_to_cb<cb_dyadd, cb_dbeta>();
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
