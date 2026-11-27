// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_stats.hpp"

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

    constexpr auto cb_x = tt::CBIndex::c_0;
    DataflowBuffer dfb_x_obj(cb_x);  // input
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    DataflowBuffer dfb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_eps = tt::CBIndex::c_2;
    DataflowBuffer dfb_eps_obj(cb_eps);  // epsilon
    constexpr auto cb_gamma = tt::CBIndex::c_3;
    DataflowBuffer dfb_gamma_obj(cb_gamma);
    constexpr auto cb_beta = tt::CBIndex::c_4;
    DataflowBuffer dfb_beta_obj(cb_beta);
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    DataflowBuffer dfb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);  // mask_w

    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer dfb_out_obj(cb_out);
    static constexpr auto cb_mean = tt::CBIndex::c_17;
    DataflowBuffer dfb_mean_obj(cb_mean);
    static constexpr auto cb_rstd = tt::CBIndex::c_18;
    DataflowBuffer dfb_rstd_obj(cb_rstd);

    static constexpr auto cb_ex = tt::CBIndex::c_24;
    DataflowBuffer dfb_ex_obj(cb_ex);  // E[x]
    constexpr auto cb_xmm = tt::CBIndex::c_25;
    DataflowBuffer dfb_xmm_obj(cb_xmm);
    constexpr auto cb_xmm2 = tt::CBIndex::c_26;
    DataflowBuffer dfb_xmm2_obj(cb_xmm2);
    constexpr auto cb_xmm2sum = tt::CBIndex::c_27;
    DataflowBuffer dfb_xmm2sum_obj(cb_xmm2sum);
    constexpr auto cb_var = tt::CBIndex::c_28;
    DataflowBuffer dfb_var_obj(cb_var);
    static constexpr auto cb_recip_std = tt::CBIndex::c_29;
    DataflowBuffer dfb_recip_std_obj(cb_recip_std);  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CBIndex::c_30;
    DataflowBuffer dfb_gamma_beta_obj(cb_gamma_beta);
    constexpr auto cb_xsum = tt::CBIndex::c_31;
    DataflowBuffer dfb_xsum_obj(cb_xsum);
    constexpr uint32_t kXmmCap = kFp32Dest ? 4u : 8u;
    constexpr auto cb_shift = tt::CBIndex::c_7;
    DataflowBuffer dfb_shift_obj(cb_shift);

    constexpr uint32_t onetile = 1;
    constexpr auto x_scalar_offset_input = ckl::input(
        cb_x,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        kDataFormatReconfig,
        ckl::TileAddressing::Offset);
    constexpr auto mask_h_scalar_offset_input = ckl::input(
        cb_mask_h,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        kDataFormatReconfig,
        ckl::TileAddressing::Offset);
    constexpr auto mask_w_scalar_offset_input = ckl::input(
        cb_mask_w,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        kDataFormatReconfig,
        ckl::TileAddressing::Offset);

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

    constexpr auto cb_zero = tt::CBIndex::c_8;
    DataflowBuffer dfb_zero_obj(cb_zero);
    tile_regs_acquire();
    ln_zero_slot(0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_zero_obj.reserve_back(1);
#if defined(FP32_DEST_ACC_EN)
    pack_reconfig_data_format(cb_zero);
#endif
    pack_tile(0, cb_zero);
    dfb_zero_obj.push_back(1);
    tile_regs_release();
    dfb_zero_obj.wait_front(1);

    constexpr auto bc_dim = is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar;
    constexpr auto shift_input =
        ckl::input(cb_shift, bc_dim, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig);
    constexpr uint32_t kTileMeanBits = get_compile_time_arg_val(11);
    constexpr uint32_t kLevel = ln_level_tiles(num_inner);
    auto ln_masked = [](uint32_t w) {
        return (do_mask_h && need_to_do_mask_h(w, origin_Ht, origin_Wt)) || (do_mask_w && ((w + 1) % origin_Wt == 0));
    };

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        dfb_x_obj.wait_front(block_size);
        constexpr auto cb_tmp = cb_ex;
        constexpr bool kTile0Masked =
            (do_mask_h && (1 % origin_Ht == 0)) || (do_mask_w && (1 % origin_Wt == 0));
        if constexpr (!kTile0Masked) {
            ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_x, cb_scaler, cb_shift, ckl::ReduceInputPolicy::NoWaitNoPop>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                [](uint32_t dst_idx) {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(dst_idx, kTileMeanBits);
                });
        } else {
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<x_scalar_offset_input>{dfb_x_obj, first_tile},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(0, origin_Ht, origin_Wt),
                    ckl::CopyTile<mask_h_scalar_offset_input, ckl::Dst::D1>{dfb_mask_h_obj, first_tile},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && (1 % origin_Wt == 0),
                    ckl::CopyTile<mask_w_scalar_offset_input, ckl::Dst::D1>{dfb_mask_w_obj, first_tile},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(
                    cb_tmp, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{dfb_ex_obj});
            ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_tmp, cb_scaler, cb_shift>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                [](uint32_t dst_idx) {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(dst_idx, kTileMeanBits);
                });
        }

        dfb_shift_obj.wait_front(onetile);

        tile_regs_acquire();
        if constexpr (kFp32Dest) {
            ln_zero_slot(kSumSlot);
        }
#if defined(FP32_DEST_ACC_EN)
        reconfig_data_format(cb_x, cb_shift);
#endif
        LnFpuSum<is_lastdim_layernorm> acc{cb_x, cb_shift, cb_mask_h, cb_mask_w, kLevel, {0, 0, 0, 0}, 0, 0};
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            if (inner_idx != 0) {
                dfb_x_obj.wait_front(block_size);
            }
            for (uint32_t j = 0; j < block_size; ++j) {
                const uint32_t w = inner_idx + j;
                acc.add(
                    j,
                    do_mask_h && need_to_do_mask_h(w, origin_Ht, origin_Wt),
                    do_mask_w && ((w + 1) % origin_Wt == 0));
            }
            dfb_x_obj.pop_front(block_size);
        }
        const uint32_t sum_slot = acc.finish();

        tile_regs_commit();
        tile_regs_wait();
        dfb_xsum_obj.reserve_back(onetile);
#if defined(FP32_DEST_ACC_EN)
        pack_reconfig_data_format(cb_xsum);
#endif
        pack_tile(sum_slot, cb_xsum);
        dfb_xsum_obj.push_back(onetile);
        tile_regs_release();

        /*
         * cb_ex
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xsum, cb_scaler, cb_ex>(ckl::ReduceInputBlockShape::single());
        dfb_ex_obj.wait_front(onetile);

        if constexpr (mean_has_value) {
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(cb_ex, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                    shift_input>{dfb_ex_obj, dfb_shift_obj},
                ckl::PackTile<ckl::output(
                    cb_mean, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                    dfb_mean_obj});
        }

        /*
         * x - E[x]
         * xmm
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_x_obj.wait_front(block_size);
            for (uint32_t j0 = 0; j0 < block_size; j0 += kXmmCap) {
                const uint32_t n = (block_size - j0) < kXmmCap ? (block_size - j0) : kXmmCap;
                tile_regs_acquire();
                ln_xmm_tiles<is_lastdim_layernorm>(cb_x, j0, n, cb_shift, cb_zero, cb_ex);
                tile_regs_commit();
                tile_regs_wait();
                dfb_xmm_obj.reserve_back(n);
#if defined(FP32_DEST_ACC_EN)
                pack_reconfig_data_format(cb_xmm);
#endif
                for (uint32_t jj = 0; jj < n; ++jj) {
                    pack_tile(jj, cb_xmm);
                }
                dfb_xmm_obj.push_back(n);
                tile_regs_release();
            }
            dfb_x_obj.pop_front(block_size);

            /*
             * mask xmm
             */
            if constexpr (do_mask_h || do_mask_w) {
                for (uint32_t j = 0; j < block_size; j++) {
                    const uint32_t w_idx = inner_idx + j;
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::CopyTile<ckl::input(
                            cb_xmm, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{
                            dfb_xmm_obj},
                        ckl::runtime_if(
                            do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt),
                            ckl::CopyTile<mask_h_scalar_offset_input, ckl::Dst::D1>{dfb_mask_h_obj, first_tile},
                            ckl::Mask<>{}),
                        ckl::runtime_if(
                            do_mask_w && (w_idx + 1) % origin_Wt == 0,
                            ckl::CopyTile<mask_w_scalar_offset_input, ckl::Dst::D1>{dfb_mask_w_obj, first_tile},
                            ckl::Mask<>{}),
                        ckl::PackTile<ckl::output(
                            cb_xmm, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                            dfb_xmm_obj});
                }  // block_size loop
            }

            ckl::eltwise_chain(
                ckl::IterationShape::tiles(block_size),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        cb_xmm,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::input(
                        cb_xmm,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::Dst::D0,
                    ckl::DestAccumulation::WholeShape>{dfb_xmm_obj, dfb_xmm_obj},
                ckl::PackTile<ckl::output(
                    cb_xmm2,
                    ckl::ReservePolicy::OneUpfront,
                    ckl::PushPolicy::OneAtEnd,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Direct,
                    ckl::DestAccumulation::WholeShape)>{dfb_xmm2_obj});
            if (inner_idx == 0) {
                ckl::copy<
                    ckl::input(cb_xmm2, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        cb_xmm2sum, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_xmm2_obj, dfb_xmm2sum_obj);
            } else {
                ckl::add<
                    ckl::input(cb_xmm2sum, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(cb_xmm2, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        cb_xmm2sum, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_xmm2sum_obj, dfb_xmm2_obj, dfb_xmm2sum_obj);
            }

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
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(cb_var, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(cb_eps, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig)>{
                dfb_var_obj, dfb_eps_obj},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                cb_recip_std, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                dfb_recip_std_obj});

        if constexpr (rstd_has_value) {
            // Write on cb_rstd.
            copy_tile_to_dfb<cb_recip_std, cb_rstd>(dfb_recip_std_obj, dfb_rstd_obj, first_tile, 0);
        } else {
            dfb_recip_std_obj.wait_front(onetile);
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_reuse = cb_xmm;
        auto& dfb_reuse_obj = dfb_xmm_obj;
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            /*
             * x - E[x]
             * cb_reuse(==cb_xmm)
             */
            dfb_x_obj.wait_front(block_size);
            for (uint32_t j0 = 0; j0 < block_size; j0 += kXmmCap) {
                const uint32_t n = (block_size - j0) < kXmmCap ? (block_size - j0) : kXmmCap;
                tile_regs_acquire();
                ln_xmm_tiles<is_lastdim_layernorm>(cb_x, j0, n, cb_shift, cb_zero, cb_ex);
                tile_regs_commit();
                tile_regs_wait();
                dfb_reuse_obj.reserve_back(n);
#if defined(FP32_DEST_ACC_EN)
                pack_reconfig_data_format(cb_reuse);
#endif
                for (uint32_t jj = 0; jj < n; ++jj) {
                    pack_tile(jj, cb_reuse);
                }
                dfb_reuse_obj.push_back(n);
                tile_regs_release();
            }
            dfb_x_obj.pop_front(block_size);

            /*
             * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
             * cb_gamma_beta_or_out
             */
            constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
            auto& dfb_gamma_beta_or_out_obj = (gamma_has_value || beta_has_value) ? dfb_gamma_beta_obj : dfb_out_obj;
            ckl::mul<
                ckl::input(
                    cb_reuse,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::input(
                    cb_recip_std,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(
                    cb_gamma_beta_or_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(block_size).block_size(block_size),
                dfb_reuse_obj,
                dfb_recip_std_obj,
                dfb_gamma_beta_or_out_obj);

            if constexpr (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                auto& dfb_outg_obj = beta_has_value ? dfb_gamma_beta_obj : dfb_out_obj;
                constexpr auto gamma_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::mul<
                    ckl::input(
                        cb_gamma_beta_or_out,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::input(
                        cb_gamma,
                        gamma_bcast,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::output(cb_outg, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                    ckl::IterationShape::tiles(block_size).block_size(block_size),
                    dfb_gamma_beta_or_out_obj,
                    dfb_gamma_obj,
                    dfb_outg_obj);
            }

            if constexpr (beta_has_value) {
                constexpr auto beta_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::add<
                    ckl::input(
                        cb_gamma_beta,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::input(
                        cb_beta,
                        beta_bcast,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::InputTileMapping::Block,
                        kDataFormatReconfig),
                    ckl::output(cb_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
                    ckl::IterationShape::tiles(block_size).block_size(block_size),
                    dfb_gamma_beta_obj,
                    dfb_beta_obj,
                    dfb_out_obj);
            }
        }  // num_inner loop
        dfb_recip_std_obj.pop_front(onetile);
        dfb_ex_obj.pop_front(onetile);
        dfb_shift_obj.pop_front(onetile);

    }  // num_rows_per_core loop
    dfb_zero_obj.pop_front(1);

    dfb_scaler_obj.pop_front(onetile);
    dfb_eps_obj.pop_front(onetile);

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    if constexpr (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
