// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool gamma_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(5) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(6) == 1;

    binary_op_init_common(tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_16);

    constexpr auto cb_dy = tt::CBIndex::c_0;         // output_grad(==dy)
    constexpr auto cb_x = tt::CBIndex::c_1;          // input(==x)
    constexpr auto cb_mean = tt::CBIndex::c_2;       // mean
    constexpr auto cb_rstd = tt::CBIndex::c_3;       // rstd
    constexpr auto cb_scaler = tt::CBIndex::c_4;     // scaler
    constexpr auto cb_n_recip_n = tt::CBIndex::c_5;  // n_recip_n
    constexpr auto cb_gamma = tt::CBIndex::c_6;      // gamma
    constexpr auto cb_mask_h_w = tt::CBIndex::c_7;   // mask_h_w

    // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    constexpr auto cb_dx = tt::CBIndex::c_16;  // input_grad(==dx)

    // y = (x - mean) * rstd
    constexpr auto cb_dycopy = tt::CBIndex::c_24;  // copy output_grad(==dycopy)
    constexpr auto cb_y = tt::CBIndex::c_25;       // output(==y)
    constexpr auto cb_dysum = tt::CBIndex::c_26;   // Sum[dy]
    constexpr auto cb_ydysum = tt::CBIndex::c_27;  // Sum[y * dy]

    constexpr auto cb_tmp1 = tt::CBIndex::c_28;  // tmp1
    constexpr auto cb_tmp2 = tt::CBIndex::c_29;  // tmp2
    constexpr auto cb_tmp3 = tt::CBIndex::c_30;  // tmp3

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader
    cb_wait_front(cb_n_recip_n, 2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);  // comes from the reader
    }

    constexpr uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        cb_wait_front(cb_mean, onetile);  // comes from the reader
        cb_wait_front(cb_rstd, onetile);  // comes from the reader

        // Compute cb_y
        // y = (x - mean) * rstd
        constexpr auto cb_dyadd = cb_tmp1;
        constexpr auto cb_ydyadd = cb_tmp2;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_xmm
            // x - mean
            constexpr auto cb_xmm = cb_tmp3;
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_xmm, onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short_with_dt(cb_x, cb_mean);
                sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short_with_dt(cb_x, cb_mean);
                sub_tiles_bcast_scalar(cb_x, cb_mean, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_xmm);

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_xmm, onetile);
            tile_regs_release();

            // Compute cb_y
            // (x - mean) * rstd and mask(optional)
            tile_regs_acquire();
            cb_wait_front(cb_xmm, onetile);
            cb_reserve_back(cb_y, onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_short_with_dt(cb_xmm, cb_rstd);
                mul_tiles_bcast_cols(cb_xmm, cb_rstd, 0, 0, dst0);
            } else {
                mul_tiles_bcast_scalar_init_short_with_dt(cb_xmm, cb_rstd);
                mul_tiles_bcast_scalar(cb_xmm, cb_rstd, 0, 0, dst0);
            }

            if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(cb_mask_h_w);
                copy_tile(cb_mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(cb_mask_h_w);
                copy_tile(cb_mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_y);

            cb_pop_front(cb_xmm, onetile);
            cb_push_back(cb_y, onetile);
            tile_regs_release();

            // Copy cb_dy to cb_dycopy
            cb_reserve_back(cb_dycopy, onetile);
            if (gamma_has_value) {
                // Compute cb_dycopy
                // dycopy = dy * gamma and mask(optional)
                tile_regs_acquire();
                cb_wait_front(cb_dy, onetile);     // comes from the reader
                cb_wait_front(cb_gamma, onetile);  // comes from the reader

                if (is_groupnorm) {
                    mul_tiles_bcast_scalar_init_short_with_dt(cb_dy, cb_gamma);
                    mul_tiles_bcast_scalar(cb_dy, cb_gamma, 0, 0, dst0);
                } else {
                    if (is_lastdim_layernorm) {
                        mul_bcast_rows_init_short_with_dt(cb_dy, cb_gamma);
                        mul_tiles_bcast_rows(cb_dy, cb_gamma, 0, 0, dst0);
                    } else {
                        mul_tiles_init_with_dt(cb_dy, cb_gamma);
                        mul_tiles(cb_dy, cb_gamma, 0, 0, dst0);
                    }
                }

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_pop_front(cb_gamma, onetile);
                cb_push_back(cb_dycopy, onetile);
                tile_regs_release();
            } else {
                // Compute cb_dycopy
                // dycopy = dy and mask(optional)
                tile_regs_acquire();
                cb_wait_front(cb_dy, onetile);  // comes from the reader

                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst0);

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_push_back(cb_dycopy, onetile);
                tile_regs_release();
            }

            // Compute cb_dyadd
            cb_wait_front(cb_dycopy, onetile);
            if (wt == 0) {
                tile_regs_acquire();
                cb_reserve_back(cb_dyadd, onetile);

                copy_tile_init_with_dt(cb_dycopy);
                copy_tile(cb_dycopy, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dyadd);

                cb_push_back(cb_dyadd, onetile);
                tile_regs_release();
            } else {
                tile_regs_acquire();
                cb_wait_front(cb_dyadd, onetile);
                cb_reserve_back(cb_dyadd, onetile);

                add_tiles_init_with_dt(cb_dyadd, cb_dycopy);
                add_tiles(cb_dyadd, cb_dycopy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dyadd);

                cb_pop_front(cb_dyadd, onetile);
                cb_push_back(cb_dyadd, onetile);
                tile_regs_release();
            }
            // We don't pop cb_dycopy here.

            // Compute cb_ydy and cb_ydyadd
            constexpr auto cb_ydy = cb_tmp3;
            // Compute cb_ydy
            tile_regs_acquire();
            cb_wait_front(cb_y, onetile);
            cb_reserve_back(cb_ydy, onetile);

            mul_tiles_init_with_dt(cb_y, cb_dycopy);
            mul_tiles(cb_y, cb_dycopy, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_ydy);

            cb_pop_front(cb_y, onetile);
            cb_pop_front(cb_dycopy, onetile);
            cb_push_back(cb_ydy, onetile);
            tile_regs_release();

            // Compute cb_ydyadd
            if (wt == 0) {
                tile_regs_acquire();
                cb_wait_front(cb_ydy, onetile);
                cb_reserve_back(cb_ydyadd, onetile);

                copy_tile_init_with_dt(cb_ydy);
                copy_tile(cb_ydy, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_ydyadd);

                cb_pop_front(cb_ydy, onetile);
                cb_push_back(cb_ydyadd, onetile);
                tile_regs_release();
            } else {
                tile_regs_acquire();
                cb_wait_front(cb_ydy, onetile);
                cb_wait_front(cb_ydyadd, onetile);
                cb_reserve_back(cb_ydyadd, onetile);

                add_tiles_init_with_dt(cb_ydyadd, cb_ydy);
                add_tiles(cb_ydyadd, cb_ydy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_ydyadd);

                cb_pop_front(cb_ydy, onetile);
                cb_pop_front(cb_ydyadd, onetile);
                cb_push_back(cb_ydyadd, onetile);
                tile_regs_release();
            }
        }  // Wt loop

        // Compute cb_dysum
        // Sum[dy]
        tile_regs_acquire();
        cb_wait_front(cb_dyadd, onetile);
        cb_reserve_back(cb_dysum, onetile);

        reduce_init_delta_with_dt(cb_dysum, cb_dyadd, cb_scaler);
        reduce_tile(cb_dyadd, cb_scaler, 0, 0, dst0);
        reduce_uninit();
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_dysum);

        cb_pop_front(cb_dyadd, onetile);
        cb_push_back(cb_dysum, onetile);
        tile_regs_release();

        // Compute cb_ydysum
        // Sum[y * dy]
        tile_regs_acquire();
        cb_wait_front(cb_ydyadd, onetile);
        cb_reserve_back(cb_ydysum, onetile);

        reduce_init_delta_with_dt(cb_ydysum, cb_ydyadd, cb_scaler);
        reduce_tile(cb_ydyadd, cb_scaler, 0, 0, dst0);
        reduce_uninit();
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_ydysum);

        cb_pop_front(cb_ydyadd, onetile);
        cb_push_back(cb_ydysum, onetile);
        tile_regs_release();

        // Compute cb_recip_nrstd
        // rstd / n -> cb_tmp3
        constexpr auto cb_recip_nrstd = cb_tmp3;
        tile_regs_acquire();
        cb_reserve_back(cb_recip_nrstd, onetile);

        if (is_lastdim_layernorm) {
            mul_bcast_cols_init_short_with_dt(cb_n_recip_n, cb_rstd);
            mul_tiles_bcast_cols(cb_n_recip_n, cb_rstd, 1, 0, dst0);
        } else {
            mul_tiles_bcast_scalar_init_short_with_dt(cb_n_recip_n, cb_rstd);
            mul_tiles_bcast_scalar(cb_n_recip_n, cb_rstd, 1, 0, dst0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_recip_nrstd);

        cb_push_back(cb_recip_nrstd, onetile);
        tile_regs_release();

        // Compute cb_dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        cb_wait_front(cb_dysum, onetile);
        cb_wait_front(cb_ydysum, onetile);
        cb_wait_front(cb_recip_nrstd, onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Copy cb_dy to cb_dycopy
            cb_reserve_back(cb_dycopy, onetile);
            if (gamma_has_value) {
                // Compute cb_dycopy
                // dycopy = dy * gamma and mask(optional)
                tile_regs_acquire();
                cb_wait_front(cb_dy, onetile);     // comes from the reader
                cb_wait_front(cb_gamma, onetile);  // comes from the reader

                if (is_groupnorm) {
                    mul_tiles_bcast_scalar_init_short_with_dt(cb_dy, cb_gamma);
                    mul_tiles_bcast_scalar(cb_dy, cb_gamma, 0, 0, dst0);
                } else {
                    if (is_lastdim_layernorm) {
                        mul_bcast_rows_init_short_with_dt(cb_dy, cb_gamma);
                        mul_tiles_bcast_rows(cb_dy, cb_gamma, 0, 0, dst0);
                    } else {
                        mul_tiles_init_with_dt(cb_dy, cb_gamma);
                        mul_tiles(cb_dy, cb_gamma, 0, 0, dst0);
                    }
                }

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_pop_front(cb_gamma, onetile);
                cb_push_back(cb_dycopy, onetile);
                tile_regs_release();
            } else {
                // Compute cb_dycopy
                // dycopy = dy and mask(optional)
                tile_regs_acquire();
                cb_wait_front(cb_dy, onetile);  // comes from the reader

                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst0);

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_h_w);
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_push_back(cb_dycopy, onetile);
                tile_regs_release();
            }

            // Compute cb_ndy
            // n * dy
            constexpr auto cb_ndy = cb_tmp1;
            tile_regs_acquire();
            cb_wait_front(cb_dycopy, onetile);
            cb_reserve_back(cb_ndy, onetile);

            mul_tiles_init_with_dt(cb_n_recip_n, cb_dycopy);
            mul_tiles(cb_n_recip_n, cb_dycopy, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_ndy);

            cb_pop_front(cb_dycopy, onetile);
            cb_push_back(cb_ndy, onetile);
            tile_regs_release();

            // Compute cb_ndymdysum
            // n * dy - Sum[dy]
            constexpr auto cb_ndymdysum = cb_tmp2;
            tile_regs_acquire();
            cb_wait_front(cb_ndy, onetile);
            cb_reserve_back(cb_ndymdysum, onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short_with_dt(cb_ndy, cb_dysum);
                sub_tiles_bcast_cols(cb_ndy, cb_dysum, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short_with_dt(cb_ndy, cb_dysum);
                sub_tiles_bcast_scalar(cb_ndy, cb_dysum, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_ndymdysum);

            cb_pop_front(cb_ndy, onetile);
            cb_push_back(cb_ndymdysum, onetile);
            tile_regs_release();

            // Compute cb_xmm
            // x - mean and mask(optional)
            constexpr auto cb_xmm = cb_tmp1;
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_xmm, onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short_with_dt(cb_x, cb_mean);
                sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short_with_dt(cb_x, cb_mean);
                sub_tiles_bcast_scalar(cb_x, cb_mean, 0, 0, dst0);
            }

            if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(cb_mask_h_w);
                copy_tile(cb_mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(cb_mask_h_w);
                copy_tile(cb_mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_xmm);

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_xmm, onetile);
            tile_regs_release();

            // Compute cb_y
            // (x - mean) * rstd
            tile_regs_acquire();
            cb_wait_front(cb_xmm, onetile);
            cb_reserve_back(cb_y, onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_short_with_dt(cb_xmm, cb_rstd);
                mul_tiles_bcast_cols(cb_xmm, cb_rstd, 0, 0, dst0);
            } else {
                mul_tiles_bcast_scalar_init_short_with_dt(cb_xmm, cb_rstd);
                mul_tiles_bcast_scalar(cb_xmm, cb_rstd, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_y);

            cb_pop_front(cb_xmm, onetile);
            cb_push_back(cb_y, onetile);
            tile_regs_release();

            // Compute cb_yydysum
            // y * Sum[y * dy]
            constexpr auto cb_yydysum = cb_tmp1;
            tile_regs_acquire();
            cb_wait_front(cb_y, onetile);
            cb_reserve_back(cb_yydysum, onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_short_with_dt(cb_y, cb_ydysum);
                mul_tiles_bcast_cols(cb_y, cb_ydysum, 0, 0, dst0);
            } else {
                mul_tiles_bcast_scalar_init_short_with_dt(cb_y, cb_ydysum);
                mul_tiles_bcast_scalar(cb_y, cb_ydysum, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_yydysum);

            cb_pop_front(cb_y, onetile);
            cb_push_back(cb_yydysum, onetile);
            tile_regs_release();

            // Compute cb_tmp4
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto cb_tmp4 = cb_y;
            tile_regs_acquire();
            cb_wait_front(cb_ndymdysum, onetile);
            cb_wait_front(cb_yydysum, onetile);
            cb_reserve_back(cb_tmp4, onetile);

            sub_tiles_init_with_dt(cb_ndymdysum, cb_yydysum);
            sub_tiles(cb_ndymdysum, cb_yydysum, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_tmp4);

            cb_pop_front(cb_ndymdysum, onetile);
            cb_pop_front(cb_yydysum, onetile);
            cb_push_back(cb_tmp4, onetile);
            tile_regs_release();

            // Compute cb_dx
            // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
            tile_regs_acquire();
            cb_wait_front(cb_tmp4, onetile);
            cb_reserve_back(cb_dx, onetile);

            mul_tiles_init_with_dt(cb_tmp4, cb_recip_nrstd);
            mul_tiles(cb_tmp4, cb_recip_nrstd, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_dx);

            cb_pop_front(cb_tmp4, onetile);
            cb_push_back(cb_dx, onetile);
            tile_regs_release();
        }  // Wt loop
        cb_pop_front(cb_recip_nrstd, onetile);
        cb_pop_front(cb_dysum, onetile);
        cb_pop_front(cb_ydysum, onetile);

        cb_pop_front(cb_mean, onetile);
        cb_pop_front(cb_rstd, onetile);
    }  // NCHt loop
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_n_recip_n, 2);

    if (do_mask_h || do_mask_w) {
        cb_pop_front(cb_mask_h_w, 2);
    }
}  // void MAIN
}  // namespace NAMESPACE
