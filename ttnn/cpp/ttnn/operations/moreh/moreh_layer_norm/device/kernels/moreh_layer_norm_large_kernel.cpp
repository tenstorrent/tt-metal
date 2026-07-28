// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

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
    DataflowBuffer dfb_x_obj(cb_x);  // input
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    DataflowBuffer dfb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_eps = tt::CBIndex::c_2;
    DataflowBuffer dfb_eps_obj(cb_eps);  // epsilon
    constexpr auto cb_gamma = tt::CBIndex::c_3;
    DataflowBuffer dfb_gamma_obj(cb_gamma);  // gamma
    constexpr auto cb_beta = tt::CBIndex::c_4;
    DataflowBuffer dfb_beta_obj(cb_beta);  // beta
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    DataflowBuffer dfb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);  // mask_w

    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer dfb_out_obj(cb_out);  // output
    constexpr auto cb_mean = tt::CBIndex::c_17;
    DataflowBuffer dfb_mean_obj(cb_mean);  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_18;
    DataflowBuffer dfb_rstd_obj(cb_rstd);  // rstd

    constexpr auto cb_ex = tt::CBIndex::c_24;
    DataflowBuffer dfb_ex_obj(cb_ex);  // E[x]
    constexpr auto cb_xmm = tt::CBIndex::c_25;
    DataflowBuffer dfb_xmm_obj(cb_xmm);  // x - E[x]
    constexpr auto cb_xmm2 = tt::CBIndex::c_26;
    DataflowBuffer dfb_xmm2_obj(cb_xmm2);  // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CBIndex::c_27;
    DataflowBuffer dfb_xmm2sum_obj(cb_xmm2sum);  // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CBIndex::c_28;
    DataflowBuffer dfb_var_obj(cb_var);  // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_std = tt::CBIndex::c_29;
    DataflowBuffer dfb_recip_std_obj(cb_recip_std);  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CBIndex::c_30;
    DataflowBuffer dfb_gamma_beta_obj(cb_gamma_beta);  // p * gamm + beta
    constexpr auto cb_xsum = tt::CBIndex::c_31;
    DataflowBuffer dfb_xsum_obj(cb_xsum);  // Sum[x]

    constexpr uint32_t onetile = 1;

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader
    dfb_eps_obj.wait_front(onetile);     // comes from the reader

    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }
    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
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
            dfb_x_obj.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                if (w_idx == 0) {
                    tile_regs_acquire();
                    dfb_xsum_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(dfb_x_obj);
                    copy_tile(cb_x, first_tile, dst0);  // input

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(dfb_mask_h_obj);
                        copy_tile(cb_mask_h, first_tile, dst1);  // mask_h
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(dfb_mask_w_obj);
                        copy_tile(cb_mask_w, first_tile, dst1);  // mask_w
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_xsum_obj);
                    dfb_xsum_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    // I use cb_ex temporarily.
                    constexpr auto cb_tmp = cb_ex;
                    DataflowBuffer dfb_tmp_obj(cb_tmp);
                    dfb_tmp_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(dfb_x_obj);
                    copy_tile(cb_x, j, j);  // input

                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(dfb_mask_h_obj);
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(dfb_mask_w_obj);
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, dfb_tmp_obj);
                    dfb_tmp_obj.push_back(onetile);
                    tile_regs_release();

                    tile_regs_acquire();
                    dfb_tmp_obj.wait_front(onetile);
                    dfb_xsum_obj.wait_front(onetile);
                    dfb_xsum_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(dfb_xsum_obj, dfb_tmp_obj);
                    add_tiles(cb_xsum, cb_tmp, first_tile, first_tile, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_xsum_obj);

                    dfb_tmp_obj.pop_front(onetile);
                    dfb_xsum_obj.pop_front(onetile);
                    dfb_xsum_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // block_size loop
            dfb_x_obj.pop_front(block_size);
        }  // num_inner loop

        /*
         * E[x]
         * cb_ex
         */
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_xsum, cb_scaler, cb_ex>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        dfb_ex_obj.wait_front(onetile);
        if (mean_has_value) {
            // Write on cb_mean.
            tile_regs_acquire();
            dfb_mean_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_ex_obj, is_lastdim_layernorm);
            copy_tile(cb_ex, first_tile, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_mean_obj);

            dfb_mean_obj.push_back(onetile);
            tile_regs_release();
        }
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_x_obj.wait_front(block_size);
            dfb_xmm_obj.reserve_back(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_short_with_dt(dfb_x_obj, dfb_ex_obj);
                    sub_tiles_bcast_cols(cb_x, cb_ex, j, first_tile, j);
                } else {
                    sub_tiles_bcast_scalar_init_short_with_dt(dfb_x_obj, dfb_ex_obj);
                    sub_tiles_bcast_scalar(cb_x, cb_ex, j, first_tile, j);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(j, dfb_xmm_obj);
                tile_regs_release();
            }  // block_size loop
            dfb_x_obj.pop_front(block_size);
            dfb_xmm_obj.push_back(block_size);

            /*
             * mask xmm
             */
            if (do_mask_h || do_mask_w) {
                dfb_xmm_obj.wait_front(block_size);
                dfb_xmm_obj.reserve_back(block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();

                    copy_tile_init_with_dt(dfb_xmm_obj);
                    copy_tile(cb_xmm, j, j);  // xmm

                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;
                    const uint32_t w_idx = inner_idx + j;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(dfb_mask_h_obj);
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    if (do_mask_w && (w_idx + 1) % origin_Wt == 0) {
                        copy_tile_init_with_dt(dfb_mask_w_obj);
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }

                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, dfb_xmm_obj);
                    tile_regs_release();
                }  // block_size loop
                dfb_xmm_obj.pop_front(block_size);
                dfb_xmm_obj.push_back(block_size);
            }

            /*
             * (x - E[x])^2
             * cb_xmm2
             */
            dfb_xmm_obj.wait_front(block_size);
            dfb_xmm2_obj.reserve_back(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                tile_regs_acquire();
                mul_tiles_init_with_dt(dfb_xmm_obj, dfb_xmm_obj);
                mul_tiles(cb_xmm, cb_xmm, j, j, j);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(j, dfb_xmm2_obj);

                tile_regs_release();
            }  // block_size loop
            dfb_xmm_obj.pop_front(block_size);
            dfb_xmm2_obj.push_back(block_size);

            /*
             * Sum[(x-E[x])^2]
             * cb_xmm2sum
             */
            dfb_xmm2_obj.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                if (inner_idx == 0 && j == 0) {
                    tile_regs_acquire();
                    dfb_xmm2sum_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(dfb_xmm2_obj);
                    copy_tile(cb_xmm2, first_tile, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_xmm2sum_obj);

                    dfb_xmm2sum_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    dfb_xmm2sum_obj.wait_front(onetile);
                    dfb_xmm2sum_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(dfb_xmm2sum_obj, dfb_xmm2_obj);
                    add_tiles(cb_xmm2sum, cb_xmm2, first_tile, j, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_xmm2sum_obj);

                    dfb_xmm2sum_obj.pop_front(onetile);
                    dfb_xmm2sum_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // block_size loop
            dfb_xmm2_obj.pop_front(block_size);
        }  // num_inner loop
        // Do not pop cb_ex here, we need it later.

        /*
         * E[(x-E[x])^2 = Var[x]
         * cb_var
         */
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_xmm2sum, cb_scaler, cb_var>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_std
         */
        tile_regs_acquire();
        dfb_var_obj.wait_front(onetile);
        dfb_recip_std_obj.reserve_back(onetile);

        add_tiles_init_with_dt(dfb_var_obj, dfb_eps_obj);
        add_tiles(cb_var, cb_eps, first_tile, first_tile, dst0);

        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_recip_std_obj);

        dfb_var_obj.pop_front(onetile);
        dfb_recip_std_obj.push_back(onetile);
        tile_regs_release();

        dfb_recip_std_obj.wait_front(onetile);
        if (rstd_has_value) {
            // Write on cb_rstd.
            tile_regs_acquire();
            dfb_rstd_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_recip_std_obj, is_lastdim_layernorm);
            copy_tile(cb_recip_std, first_tile, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_rstd_obj);

            dfb_rstd_obj.push_back(onetile);
            tile_regs_release();
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_reuse = cb_xmm;
        DataflowBuffer dfb_reuse_obj(cb_reuse);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            /*
             * x - E[x]
             * cb_reuse(==cb_xmm)
             */
            dfb_x_obj.wait_front(block_size);
            dfb_reuse_obj.reserve_back(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_short_with_dt(dfb_x_obj, dfb_ex_obj);
                    sub_tiles_bcast_cols(cb_x, cb_ex, j, first_tile, j);
                } else {
                    sub_tiles_bcast_scalar_init_short_with_dt(dfb_x_obj, dfb_ex_obj);
                    sub_tiles_bcast_scalar(cb_x, cb_ex, j, first_tile, j);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(j, dfb_reuse_obj);
                tile_regs_release();
            }  // block_size loop
            dfb_x_obj.pop_front(block_size);
            dfb_reuse_obj.push_back(block_size);

            /*
             * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
             * cb_gamma_beta_or_out
             */
            constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
            DataflowBuffer dfb_gamma_beta_or_out_obj(cb_gamma_beta_or_out);
            dfb_reuse_obj.wait_front(block_size);
            dfb_gamma_beta_or_out_obj.reserve_back(block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    mul_bcast_cols_init_short_with_dt(dfb_reuse_obj, dfb_recip_std_obj);
                    mul_tiles_bcast_cols(cb_reuse, cb_recip_std, j, first_tile, j);
                } else {
                    mul_tiles_bcast_scalar_init_short_with_dt(dfb_reuse_obj, dfb_recip_std_obj);
                    mul_tiles_bcast_scalar(cb_reuse, cb_recip_std, j, first_tile, j);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(j, dfb_gamma_beta_or_out_obj);
                tile_regs_release();
            }  // block_size loop
            dfb_reuse_obj.pop_front(block_size);
            dfb_gamma_beta_or_out_obj.push_back(block_size);

            // * gamma
            if (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                DataflowBuffer dfb_outg_obj(cb_outg);
                dfb_gamma_beta_or_out_obj.wait_front(block_size);
                dfb_gamma_obj.wait_front(block_size);
                dfb_outg_obj.reserve_back(block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();
                    if (is_groupnorm) {
                        mul_tiles_bcast_scalar_init_short_with_dt(dfb_gamma_beta_or_out_obj, dfb_gamma_obj);
                        mul_tiles_bcast_scalar(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            mul_bcast_rows_init_short_with_dt(dfb_gamma_beta_or_out_obj, dfb_gamma_obj);
                            mul_tiles_bcast_rows(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        } else {
                            mul_tiles_init_with_dt(dfb_gamma_beta_or_out_obj, dfb_gamma_obj);
                            mul_tiles(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        }
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, dfb_outg_obj);
                    tile_regs_release();
                }  // block_size loop
                dfb_gamma_beta_or_out_obj.pop_front(block_size);
                dfb_gamma_obj.pop_front(block_size);
                dfb_outg_obj.push_back(block_size);
            }

            // + beta
            if (beta_has_value) {
                dfb_gamma_beta_obj.wait_front(block_size);
                dfb_beta_obj.wait_front(block_size);
                dfb_out_obj.reserve_back(block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();
                    if (is_groupnorm) {
                        add_bcast_scalar_init_short_with_dt(dfb_gamma_beta_obj, dfb_beta_obj);
                        add_tiles_bcast_scalar(cb_gamma_beta, cb_beta, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            add_bcast_rows_init_short_with_dt(dfb_gamma_beta_obj, dfb_beta_obj);
                            add_tiles_bcast_rows(cb_gamma_beta, cb_beta, j, j, j);
                        } else {
                            add_tiles_init_with_dt(dfb_gamma_beta_obj, dfb_beta_obj);
                            add_tiles(cb_gamma_beta, cb_beta, j, j, j);
                        }
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, dfb_out_obj);
                    tile_regs_release();
                }  // block_size loop
                dfb_gamma_beta_obj.pop_front(block_size);
                dfb_beta_obj.pop_front(block_size);
                dfb_out_obj.push_back(block_size);
            }
        }  // num_inner loop
        dfb_recip_std_obj.pop_front(onetile);
        dfb_ex_obj.pop_front(onetile);
    }  // num_rows_per_core loop
    dfb_scaler_obj.pop_front(onetile);
    dfb_eps_obj.pop_front(onetile);

    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
