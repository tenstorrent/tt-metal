// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add/sub/mul
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // Rsqrt

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

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    constexpr auto dfb_x_id = tt::CBIndex::c_0;
    DataflowBuffer dfb_x_obj(dfb_x_id);  // input
    constexpr auto dfb_scaler_id = tt::CBIndex::c_1;
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);  // scaler
    constexpr auto dfb_eps_id = tt::CBIndex::c_2;
    DataflowBuffer dfb_eps_obj(dfb_eps_id);  // epsilon
    constexpr auto dfb_gamma_id = tt::CBIndex::c_3;
    constexpr auto dfb_beta_id = tt::CBIndex::c_4;
    constexpr auto dfb_mask_h_id = tt::CBIndex::c_5;
    DataflowBuffer dfb_mask_h_obj(dfb_mask_h_id);  // mask_h
    constexpr auto dfb_mask_w_id = tt::CBIndex::c_6;
    DataflowBuffer dfb_mask_w_obj(dfb_mask_w_id);  // mask_w

    constexpr auto dfb_out_id = tt::CBIndex::c_16;
    constexpr auto dfb_mean_id = tt::CBIndex::c_17;
    constexpr auto dfb_rstd_id = tt::CBIndex::c_18;

    constexpr auto dfb_ex_id = tt::CBIndex::c_24;
    DataflowBuffer dfb_ex_obj(dfb_ex_id);  // E[x]
    constexpr auto dfb_xmm_id = tt::CBIndex::c_25;
    constexpr auto dfb_xmm2_id = tt::CBIndex::c_26;
    constexpr auto dfb_xmm2sum_id = tt::CBIndex::c_27;
    constexpr auto dfb_var_id = tt::CBIndex::c_28;
    constexpr auto dfb_recip_std_id = tt::CBIndex::c_29;
    DataflowBuffer dfb_recip_std_obj(dfb_recip_std_id);  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto dfb_gamma_beta_id = tt::CBIndex::c_30;
    constexpr auto dfb_xsum_id = tt::CBIndex::c_31;

    constexpr uint32_t onetile = 1;

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader
    dfb_eps_obj.wait_front(onetile);     // comes from the reader

    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }
    if constexpr (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    constexpr uint32_t first_tile = 0;

    constexpr uint32_t origin_Ht = (origin_H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    constexpr uint32_t origin_Wt = (origin_W + TILE_WIDTH - 1) / TILE_WIDTH;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        /*
         * Sum[x]
         * dfb_xsum_id
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_x_obj.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                if (w_idx == 0) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::CopyTile<ckl::input(
                            dfb_x_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Scalar,
                            kDataFormatReconfig,
                            ckl::TileAddressing::Offset)>{first_tile},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_h_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::runtime_if(
                            do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_w_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(
                            dfb_xsum_id,
                            ckl::ReservePolicy::PerTile,
                            ckl::PushPolicy::PerTile,
                            kDataFormatReconfig)>{});
                } else {
                    // I use dfb_ex_id temporarily.
                    constexpr auto dfb_tmp_id = dfb_ex_id;
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::CopyTile<ckl::input(
                            dfb_x_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Scalar,
                            kDataFormatReconfig,
                            ckl::TileAddressing::Offset)>{j},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_h_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::runtime_if(
                            do_mask_w && ((w_idx + 1) % origin_Wt == 0),
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_w_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(
                            dfb_tmp_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Add,
                            ckl::input(
                                dfb_xsum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                            ckl::input(
                                dfb_tmp_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                        ckl::PackTile<ckl::output(
                            dfb_xsum_id,
                            ckl::ReservePolicy::PerTile,
                            ckl::PushPolicy::PerTile,
                            kDataFormatReconfig)>{});
                }
            }  // block_size loop
            dfb_x_obj.pop_front(block_size);
        }  // num_inner loop

        /*
         * E[x]
         * dfb_ex_id
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_xsum_id, dfb_scaler_id, dfb_ex_id>(ckl::ReduceInputBlockShape::single());

        if constexpr (mean_has_value) {
            // Write on dfb_mean_id.
            copy_tile_to_dfb<dfb_ex_id, dfb_mean_id>(first_tile, 0);
        } else {
            dfb_ex_obj.wait_front(onetile);
        }
        // We don't pop dfb_ex_id here.

        /*
         * x - E[x]
         * xmm
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            ckl::sub<
                ckl::input(
                    dfb_x_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::input(
                    dfb_ex_id,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_xmm_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(block_size).block_size(block_size));

            /*
             * mask xmm
             */
            if constexpr (do_mask_h || do_mask_w) {
                for (uint32_t j = 0; j < block_size; j++) {
                    const uint32_t w_idx = inner_idx + j;
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::CopyTile<ckl::input(
                            dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_h_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::runtime_if(
                            do_mask_w && (w_idx + 1) % origin_Wt == 0,
                            ckl::CopyTile<
                                ckl::input(
                                    dfb_mask_w_id,
                                    ckl::WaitPolicy::None,
                                    ckl::PopPolicy::None,
                                    ckl::InputTileMapping::Scalar,
                                    kDataFormatReconfig,
                                    ckl::TileAddressing::Offset),
                                ckl::Dst::D1>{first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(
                            dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});
                }  // block_size loop
            }

            /*
             * (x - E[x])^2
             * dfb_xmm2_id
             */
            ckl::square<
                ckl::input(
                    dfb_xmm_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::output(dfb_xmm2_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(block_size).block_size(block_size));

            /*
             * Sum[(x-E[x])^2]
             * dfb_xmm2sum_id
             */
            for (uint32_t j = 0; j < block_size; j++) {
                if (inner_idx == 0 && j == 0) {
                    ckl::copy<
                        ckl::input(dfb_xmm2_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::output(
                            dfb_xmm2sum_id,
                            ckl::ReservePolicy::PerTile,
                            ckl::PushPolicy::PerTile,
                            kDataFormatReconfig)>(ckl::IterationShape::one_tile());
                } else {
                    ckl::add<
                        ckl::input(
                            dfb_xmm2sum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::input(dfb_xmm2_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                        ckl::output(
                            dfb_xmm2sum_id,
                            ckl::ReservePolicy::PerTile,
                            ckl::PushPolicy::PerTile,
                            kDataFormatReconfig)>(ckl::IterationShape::one_tile());
                }
            }  // block_size loop
        }  // num_inner loop
        // Do not pop dfb_ex_id here, we need it later.

        /*
         * E[(x-E[x])^2 = Var[x]
         * dfb_var_id
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_xmm2sum_id, dfb_scaler_id, dfb_var_id>(
            ckl::ReduceInputBlockShape::single());

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * dfb_recip_std_id
         */
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_var_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig)>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_recip_std_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        if constexpr (rstd_has_value) {
            // Write on dfb_rstd_id.
            copy_tile_to_dfb<dfb_recip_std_id, dfb_rstd_id>(first_tile, 0);
        } else {
            dfb_recip_std_obj.wait_front(onetile);
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * dfb_out_id
         */
        constexpr auto dfb_reuse_id = dfb_xmm_id;
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            /*
             * x - E[x]
             * dfb_reuse_id(==dfb_xmm_id)
             */
            ckl::sub<
                ckl::input(
                    dfb_x_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::input(
                    dfb_ex_id,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_reuse_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(block_size).block_size(block_size));

            /*
             * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
             * dfb_gamma_beta_or_out_id
             */
            constexpr auto dfb_gamma_beta_or_out_id =
                (gamma_has_value || beta_has_value) ? dfb_gamma_beta_id : dfb_out_id;
            ckl::mul<
                ckl::input(
                    dfb_reuse_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::input(
                    dfb_recip_std_id,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(
                    dfb_gamma_beta_or_out_id,
                    ckl::ReservePolicy::Upfront,
                    ckl::PushPolicy::AtEnd,
                    kDataFormatReconfig)>(ckl::IterationShape::tiles(block_size).block_size(block_size));

            if constexpr (gamma_has_value) {
                constexpr auto dfb_outg_id = beta_has_value ? dfb_gamma_beta_id : dfb_out_id;
                constexpr auto gamma_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::mul<
                    ckl::input(
                        dfb_gamma_beta_or_out_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::input(
                        dfb_gamma_id,
                        gamma_bcast,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::output(dfb_outg_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                    ckl::IterationShape::tiles(block_size).block_size(block_size));
            }

            if constexpr (beta_has_value) {
                constexpr auto beta_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::add<
                    ckl::input(
                        dfb_gamma_beta_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::input(
                        dfb_beta_id,
                        beta_bcast,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::output(dfb_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                    ckl::IterationShape::tiles(block_size).block_size(block_size));
            }
        }  // num_inner loop
        dfb_recip_std_obj.pop_front(onetile);
        dfb_ex_obj.pop_front(onetile);
    }  // num_rows_per_core loop
    dfb_scaler_obj.pop_front(onetile);
    dfb_eps_obj.pop_front(onetile);

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    if constexpr (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
