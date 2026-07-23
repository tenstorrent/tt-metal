// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // add/sub/mul
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Rsqrt

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
    constexpr uint32_t num_inner = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(7) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(8) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(9) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(10) == 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    constexpr auto cb_x = tt::CBIndex::c_0;
    DataflowBuffer cb_x_obj(cb_x);  // input
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    DataflowBuffer cb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_eps = tt::CBIndex::c_2;
    DataflowBuffer cb_eps_obj(cb_eps);  // epsilon
    constexpr auto cb_gamma = tt::CBIndex::c_3;
    constexpr auto cb_beta = tt::CBIndex::c_4;
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    DataflowBuffer cb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    DataflowBuffer cb_mask_w_obj(cb_mask_w);  // mask_w

    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_mean = tt::CBIndex::c_17;
    constexpr auto cb_rstd = tt::CBIndex::c_18;

    constexpr auto cb_ex = tt::CBIndex::c_24;
    DataflowBuffer cb_ex_obj(cb_ex);  // E[x]
    constexpr auto cb_xmm = tt::CBIndex::c_25;
    constexpr auto cb_xmm2 = tt::CBIndex::c_26;
    constexpr auto cb_xmm2sum = tt::CBIndex::c_27;
    constexpr auto cb_var = tt::CBIndex::c_28;
    constexpr auto cb_recip_std = tt::CBIndex::c_29;
    DataflowBuffer cb_recip_std_obj(cb_recip_std);  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CBIndex::c_30;
    constexpr auto cb_xsum = tt::CBIndex::c_31;

    constexpr uint32_t onetile = 1;

    cb_scaler_obj.wait_front(onetile);  // comes from the reader
    cb_eps_obj.wait_front(onetile);     // comes from the reader

    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    if constexpr (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
    }
    if constexpr (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
    }

    constexpr uint32_t first_tile = 0;

    constexpr uint32_t origin_Ht = (origin_H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    constexpr uint32_t origin_Wt = (origin_W + TILE_WIDTH - 1) / TILE_WIDTH;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        /*
         * Sum[x]
         * cb_xsum
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_x_obj.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                if (w_idx == 0) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::single(),
                        ckl::CopyTile<ckl::input(
                            cb_x,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set)>{first_tile},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    cb_mask_h,
                                    ckl::InputLifecycle::CallerManaged,
                                    ckl::OperandKind::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileOffset::Set),
                                ckl::Dst::D1>{first_tile},
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
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(cb_xsum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
                } else {
                    // I use cb_ex temporarily.
                    constexpr auto cb_tmp = cb_ex;
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::single(),
                        ckl::CopyTile<ckl::input(
                            cb_x,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set)>{j},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    cb_mask_h,
                                    ckl::InputLifecycle::CallerManaged,
                                    ckl::OperandKind::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileOffset::Set),
                                ckl::Dst::D1>{first_tile},
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
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(cb_tmp, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

                    ckl::eltwise_chain(
                        ckl::EltwiseShape::single(),
                        ckl::BinaryFpu<
                            ckl::input(cb_xsum, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                            ckl::input(cb_tmp, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                            ckl::BinaryFpuOp::Add,
                            ckl::BroadcastDim::None>{},
                        ckl::PackTile<ckl::output(cb_xsum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
                }
            }  // block_size loop
            cb_x_obj.pop_front(block_size);
        }  // num_inner loop

        /*
         * E[x]
         * cb_ex
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xsum, cb_scaler, cb_ex>(ckl::ReduceInputBlockShape::single());

        if constexpr (mean_has_value) {
            copy_tile_to_cb<cb_ex, cb_mean>(first_tile, 0);
        } else {
            cb_ex_obj.wait_front(onetile);
        }
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            ckl::sub<
                ckl::input(cb_x, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                ckl::input(cb_ex, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_xmm, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * mask xmm
             */
            if constexpr (do_mask_h || do_mask_w) {
                for (uint32_t j = 0; j < block_size; j++) {
                    const uint32_t w_idx = inner_idx + j;
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::single(),
                        ckl::CopyTile<ckl::input(cb_xmm, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    cb_mask_h,
                                    ckl::InputLifecycle::CallerManaged,
                                    ckl::OperandKind::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileOffset::Set),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::runtime_if(
                            do_mask_w && (w_idx + 1) % origin_Wt == 0,
                            ckl::CopyTile<
                                ckl::input(
                                    cb_mask_w,
                                    ckl::InputLifecycle::CallerManaged,
                                    ckl::OperandKind::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileOffset::Set),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(cb_xmm, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
                }  // block_size loop
            }

            /*
             * (x - E[x])^2
             * cb_xmm2
             */
            ckl::square<
                ckl::input(cb_xmm, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                ckl::output(cb_xmm2, ckl::OutputLifecycle::Bulk, kDataFormatReconfig)>(
                ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * Sum[(x-E[x])^2]
             * cb_xmm2sum
             */
            for (uint32_t j = 0; j < block_size; j++) {
                if (inner_idx == 0 && j == 0) {
                    ckl::copy<
                        ckl::input(cb_xmm2, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::output(cb_xmm2sum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                        ckl::EltwiseShape::single());
                } else {
                    ckl::add<
                        ckl::input(cb_xmm2sum, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::input(cb_xmm2, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::output(cb_xmm2sum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                        ckl::EltwiseShape::single());
                }
            }  // block_size loop
        }  // num_inner loop
        // Do not pop cb_ex here, we need it later.

        /*
         * E[(x-E[x])^2 = Var[x]
         * cb_var
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xmm2sum, cb_scaler, cb_var>(ckl::ReduceInputBlockShape::single());

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_std
         */
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_var, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_eps, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_std, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        if constexpr (rstd_has_value) {
            copy_tile_to_cb<cb_recip_std, cb_rstd>(first_tile, 0);
        } else {
            cb_recip_std_obj.wait_front(onetile);
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_reuse = cb_xmm;
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            /*
             * x - E[x]
             * cb_reuse(==cb_xmm)
             */
            ckl::sub<
                ckl::input(cb_x, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                ckl::input(cb_ex, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_reuse, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
             * cb_gamma_beta_or_out
             */
            constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
            ckl::mul<
                ckl::input(cb_reuse, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                ckl::input(cb_recip_std, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_gamma_beta_or_out, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(block_size, block_size));

            if constexpr (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                constexpr auto gamma_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::mul<
                    ckl::input(
                        cb_gamma_beta_or_out, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                    ckl::input(cb_gamma, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                    ckl::output(cb_outg, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
                    gamma_bcast>(ckl::EltwiseShape::tiles(block_size, block_size));
            }

            if constexpr (beta_has_value) {
                constexpr auto beta_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::add<
                    ckl::input(cb_gamma_beta, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                    ckl::input(cb_beta, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
                    ckl::output(cb_out, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
                    beta_bcast>(ckl::EltwiseShape::tiles(block_size, block_size));
            }
        }  // num_inner loop
        cb_recip_std_obj.pop_front(onetile);
        cb_ex_obj.pop_front(onetile);
    }  // num_rows_per_core loop
    cb_scaler_obj.pop_front(onetile);
    cb_eps_obj.pop_front(onetile);

    if constexpr (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
    if constexpr (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
}
