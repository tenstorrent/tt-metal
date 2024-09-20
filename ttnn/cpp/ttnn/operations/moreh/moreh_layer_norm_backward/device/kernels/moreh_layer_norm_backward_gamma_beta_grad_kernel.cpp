// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"


namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_cols_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t NCHt = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_grad_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(7) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(8) == 1;

    constexpr auto cb_dy = tt::CB::c_in0;      // output_grad(==dy)
    constexpr auto cb_x = tt::CB::c_in1;       // input(==x)
    constexpr auto cb_mean = tt::CB::c_in2;    // mean
    constexpr auto cb_rstd = tt::CB::c_in3;    // rstd
    constexpr auto cb_scaler = tt::CB::c_in4;  // scaler
    constexpr auto cb_mask_h = tt::CB::c_in5;  // mask_h
    constexpr auto cb_mask_w = tt::CB::c_in6;  // mask_w

    // Sum[y * dy]
    constexpr auto cb_dgamma = tt::CB::c_out0;  // gamma_grad(==dgamma)
    // Sum[dy]
    constexpr auto cb_dbeta = tt::CB::c_out1;  // beta_grad(==dbeta)

    // y = (x - mean) * rstd
    constexpr auto cb_y = tt::CB::c_intermed0;       // output(==y)
    constexpr auto cb_ydy = tt::CB::c_intermed1;     // y * dy
    constexpr auto cb_dyadd = tt::CB::c_intermed2;   // Add[dy]
    constexpr auto cb_ydyadd = tt::CB::c_intermed3;  // Add[y * dy]
    constexpr auto cb_xmm = tt::CB::c_intermed4;     // x - mean
    constexpr auto cb_dycopy = tt::CB::c_intermed5;  // dycopy

    constexpr uint32_t onetile = 1;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm);
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t Ht = origin_Ht;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    constexpr uint32_t HtWt = Ht * Wt;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    cb_wait_front(cb_scaler, onetile);  // comes from the reader

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    uint32_t h_idx;
    uint32_t w_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_cols_per_core; outer_idx++) {
        for (uint32_t inner_idx = 0; inner_idx < NCHt; inner_idx++) {
            if (is_groupnorm) {
                h_idx = (inner_idx % HtWt) / Wt;
                w_idx = (inner_idx % HtWt) % Wt;
            } else {
                h_idx = inner_idx;
                w_idx = outer_idx;
            }

            // Compute cb_dycopy
            // deepcopy and mask(optional)
            tile_regs_acquire();
            cb_wait_front(cb_dy, onetile);  // comes from the reader
            cb_reserve_back(cb_dycopy, onetile);

            copy_tile_init_with_dt(cb_dy);
            copy_tile(cb_dy, 0, dst0);

            if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                copy_tile_init_with_dt(cb_mask_h);
                copy_tile(cb_mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(cb_mask_w);
                copy_tile(cb_mask_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_dycopy);

            cb_pop_front(cb_dy, onetile);
            cb_push_back(cb_dycopy, onetile);
            tile_regs_release();

            // Compute cb_dyadd
            cb_wait_front(cb_dycopy, onetile);
            if (beta_grad_has_value) {
                if (inner_idx == 0) {
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
            }  // beta_grad_has_value
            // We don't pop cb_dycopy here.

            if (gamma_grad_has_value) {
                // Compute cb_xmm
                // x - mean and mask(optional)
                tile_regs_acquire();
                cb_wait_front(cb_x, onetile);     // comes from the reader
                cb_wait_front(cb_mean, onetile);  // comes from the reader
                cb_reserve_back(cb_xmm, onetile);

                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_short_with_dt(cb_x, cb_mean);
                    sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);
                } else {
                    sub_tiles_bcast_scalar_init_short_with_dt(cb_x, cb_mean);
                    sub_tiles_bcast_scalar(cb_x, cb_mean, 0, 0, dst0);
                }

                if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                    copy_tile_init_with_dt(cb_mask_h);
                    copy_tile(cb_mask_h, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_w);
                    copy_tile(cb_mask_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xmm);

                cb_pop_front(cb_x, onetile);
                cb_pop_front(cb_mean, onetile);
                cb_push_back(cb_xmm, onetile);
                tile_regs_release();

                // Compute cb_y
                // (x - mean) * rstd
                tile_regs_acquire();
                cb_wait_front(cb_xmm, onetile);
                cb_wait_front(cb_rstd, onetile);  // comes from the reader
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
                cb_pop_front(cb_rstd, onetile);
                cb_push_back(cb_y, onetile);
                tile_regs_release();

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
                cb_push_back(cb_ydy, onetile);
                tile_regs_release();

                // Compute cb_ydyadd
                if (inner_idx == 0) {
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
            }  // gamma_grad_has_value

            cb_pop_front(cb_dycopy, onetile);
        }  // inner_idx loop

        if (gamma_grad_has_value) {
            // Compute cb_dgamma
            tile_regs_acquire();
            cb_wait_front(cb_ydyadd, onetile);
            cb_reserve_back(cb_dgamma, onetile);

            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                reduce_init_delta_with_dt<false>(cb_dgamma, cb_ydyadd, cb_scaler);
                reduce_tile(cb_ydyadd, cb_scaler, 0, 0, dst0);
                reduce_revert_delta(cb_dgamma);
            } else {
                // Just copy
                copy_tile_init_with_dt(cb_ydyadd);
                copy_tile(cb_ydyadd, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_dgamma);

            cb_pop_front(cb_ydyadd, onetile);
            cb_push_back(cb_dgamma, onetile);
            tile_regs_release();
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // Compute cb_dbeta
            tile_regs_acquire();
            cb_wait_front(cb_dyadd, onetile);
            cb_reserve_back(cb_dbeta, onetile);

            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                reduce_init_delta_with_dt<false>(cb_dbeta, cb_dyadd, cb_scaler);
                reduce_tile(cb_dyadd, cb_scaler, 0, 0, dst0);
                reduce_revert_delta(cb_dbeta);
            } else {
                // Just copy
                copy_tile_init_with_dt(cb_dyadd);
                copy_tile(cb_dyadd, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_dbeta);

            cb_pop_front(cb_dyadd, onetile);
            cb_push_back(cb_dbeta, onetile);
            tile_regs_release();
        }  // beta_grad_has_value

    }  // outer_idx loop
    cb_pop_front(cb_scaler, onetile);

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }

}  // void MAIN
}  // namespace NAMESPACE
