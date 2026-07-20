// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // add/sub/mul
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Rsqrt

namespace ckl = compute_kernel_lib;

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
    CircularBuffer cb_x_obj(cb_x);  // input
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    CircularBuffer cb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_eps = tt::CBIndex::c_2;
    CircularBuffer cb_eps_obj(cb_eps);  // epsilon
    constexpr auto cb_gamma = tt::CBIndex::c_3;
    CircularBuffer cb_gamma_obj(cb_gamma);  // gamma
    constexpr auto cb_beta = tt::CBIndex::c_4;
    CircularBuffer cb_beta_obj(cb_beta);  // beta
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    CircularBuffer cb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    CircularBuffer cb_mask_w_obj(cb_mask_w);  // mask_w

    constexpr auto cb_out = tt::CBIndex::c_16;
    CircularBuffer cb_out_obj(cb_out);  // output
    constexpr auto cb_mean = tt::CBIndex::c_17;
    CircularBuffer cb_mean_obj(cb_mean);  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_18;
    CircularBuffer cb_rstd_obj(cb_rstd);  // rstd

    constexpr auto cb_ex = tt::CBIndex::c_24;
    CircularBuffer cb_ex_obj(cb_ex);  // E[x]
    constexpr auto cb_xmm = tt::CBIndex::c_25;
    CircularBuffer cb_xmm_obj(cb_xmm);  // x - E[x]
    constexpr auto cb_xmm2 = tt::CBIndex::c_26;
    CircularBuffer cb_xmm2_obj(cb_xmm2);  // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CBIndex::c_27;
    CircularBuffer cb_xmm2sum_obj(cb_xmm2sum);  // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CBIndex::c_28;
    CircularBuffer cb_var_obj(cb_var);  // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_std = tt::CBIndex::c_29;
    CircularBuffer cb_recip_std_obj(cb_recip_std);  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CBIndex::c_30;
    CircularBuffer cb_gamma_beta_obj(cb_gamma_beta);  // p * gamm + beta
    constexpr auto cb_xsum = tt::CBIndex::c_31;
    CircularBuffer cb_xsum_obj(cb_xsum);  // Sum[x]

    constexpr uint32_t onetile = 1;

    cb_scaler_obj.wait_front(onetile);  // comes from the reader
    cb_eps_obj.wait_front(onetile);     // comes from the reader

    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    if (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
    }
    if (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
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
                    tile_regs_acquire();
                    cb_xsum_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(DataflowBuffer(cb_x));
                    copy_tile(cb_x, first_tile, dst0);  // input

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_h));
                        copy_tile(cb_mask_h, first_tile, dst1);  // mask_h
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_w));
                        copy_tile(cb_mask_w, first_tile, dst1);  // mask_w
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, DataflowBuffer(cb_xsum));
                    cb_xsum_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    // I use cb_ex temporarily.
                    constexpr auto cb_tmp = cb_ex;
                    CircularBuffer cb_tmp_obj(cb_tmp);
                    cb_tmp_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(DataflowBuffer(cb_x));
                    copy_tile(cb_x, j, j);  // input

                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_h));
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_w));
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, DataflowBuffer(cb_tmp));
                    cb_tmp_obj.push_back(onetile);
                    tile_regs_release();

                    tile_regs_acquire();
                    cb_tmp_obj.wait_front(onetile);
                    cb_xsum_obj.wait_front(onetile);
                    cb_xsum_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(DataflowBuffer(cb_xsum), DataflowBuffer(cb_tmp));
                    add_tiles(cb_xsum, cb_tmp, first_tile, first_tile, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, DataflowBuffer(cb_xsum));

                    cb_tmp_obj.pop_front(onetile);
                    cb_xsum_obj.pop_front(onetile);
                    cb_xsum_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // block_size loop
            cb_x_obj.pop_front(block_size);
        }  // num_inner loop

        /*
         * E[x]
         * cb_ex
         */
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_xsum, cb_scaler, cb_ex>(ckl::ReduceInputBlockShape::single());

        cb_ex_obj.wait_front(onetile);
        if (mean_has_value) {
            // Write on cb_mean.
            tile_regs_acquire();
            cb_mean_obj.reserve_back(onetile);

            copy_tile_init_with_dt(DataflowBuffer(cb_ex), is_lastdim_layernorm);
            copy_tile(cb_ex, first_tile, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, DataflowBuffer(cb_mean));

            cb_mean_obj.push_back(onetile);
            tile_regs_release();
        }
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            ckl::sub<
                cb_x,
                cb_ex,
                cb_xmm,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * mask xmm
             */
            if (do_mask_h || do_mask_w) {
                cb_xmm_obj.wait_front(block_size);
                cb_xmm_obj.reserve_back(block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();

                    copy_tile_init_with_dt(DataflowBuffer(cb_xmm));
                    copy_tile(cb_xmm, j, j);  // xmm

                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;
                    const uint32_t w_idx = inner_idx + j;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_h));
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    if (do_mask_w && (w_idx + 1) % origin_Wt == 0) {
                        copy_tile_init_with_dt(DataflowBuffer(cb_mask_w));
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, DataflowBuffer(cb_xmm));
                    tile_regs_release();
                }  // block_size loop
                cb_xmm_obj.pop_front(block_size);
                cb_xmm_obj.push_back(block_size);
            }

            /*
             * (x - E[x])^2
             * cb_xmm2
             */
            ckl::square<
                cb_xmm,
                cb_xmm2,
                ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * Sum[(x-E[x])^2]
             * cb_xmm2sum
             */
            cb_xmm2_obj.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                if (inner_idx == 0 && j == 0) {
                    tile_regs_acquire();
                    cb_xmm2sum_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(DataflowBuffer(cb_xmm2));
                    copy_tile(cb_xmm2, first_tile, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, DataflowBuffer(cb_xmm2sum));

                    cb_xmm2sum_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    cb_xmm2sum_obj.wait_front(onetile);
                    cb_xmm2sum_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(DataflowBuffer(cb_xmm2sum), DataflowBuffer(cb_xmm2));
                    add_tiles(cb_xmm2sum, cb_xmm2, first_tile, j, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, DataflowBuffer(cb_xmm2sum));

                    cb_xmm2sum_obj.pop_front(onetile);
                    cb_xmm2sum_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // block_size loop
            cb_xmm2_obj.pop_front(block_size);
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
                cb_var,
                cb_eps,
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<cb_recip_std>{});

        cb_recip_std_obj.wait_front(onetile);
        if (rstd_has_value) {
            // Write on cb_rstd.
            tile_regs_acquire();
            cb_rstd_obj.reserve_back(onetile);

            copy_tile_init_with_dt(DataflowBuffer(cb_recip_std), is_lastdim_layernorm);
            copy_tile(cb_recip_std, first_tile, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, DataflowBuffer(cb_rstd));

            cb_rstd_obj.push_back(onetile);
            tile_regs_release();
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_reuse = cb_xmm;
        CircularBuffer cb_reuse_obj(cb_reuse);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            /*
             * x - E[x]
             * cb_reuse(==cb_xmm)
             */
            ckl::sub<
                cb_x,
                cb_ex,
                cb_reuse,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));

            /*
             * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
             * cb_gamma_beta_or_out
             */
            constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
            CircularBuffer cb_gamma_beta_or_out_obj(cb_gamma_beta_or_out);
            ckl::mul<
                cb_reuse,
                cb_recip_std,
                cb_gamma_beta_or_out,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));

            if (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                constexpr auto gamma_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::mul<
                    cb_gamma_beta_or_out,
                    cb_gamma,
                    cb_outg,
                    gamma_bcast,
                    ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));
            }

            if (beta_has_value) {
                constexpr auto beta_bcast =
                    is_groupnorm ? ckl::BroadcastDim::Scalar
                                 : (is_lastdim_layernorm ? ckl::BroadcastDim::Row : ckl::BroadcastDim::None);
                ckl::add<
                    cb_gamma_beta,
                    cb_beta,
                    cb_out,
                    beta_bcast,
                    ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(block_size, block_size));
            }
        }  // num_inner loop
        cb_recip_std_obj.pop_front(onetile);
        cb_ex_obj.pop_front(onetile);
    }  // num_rows_per_core loop
    cb_scaler_obj.pop_front(onetile);
    cb_eps_obj.pop_front(onetile);

    if (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
    if (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
}
