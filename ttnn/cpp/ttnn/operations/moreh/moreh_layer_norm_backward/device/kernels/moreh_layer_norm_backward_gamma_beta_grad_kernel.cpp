// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

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
    experimental::CircularBuffer cb_dy_obj(cb_dy);  // output_grad(==dy)
    constexpr auto cb_x = tt::CBIndex::c_1;
    experimental::CircularBuffer cb_x_obj(cb_x);  // input(==x)
    constexpr auto cb_mean = tt::CBIndex::c_2;
    experimental::CircularBuffer cb_mean_obj(cb_mean);  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_3;
    experimental::CircularBuffer cb_rstd_obj(cb_rstd);  // rstd
    constexpr auto cb_scaler = tt::CBIndex::c_4;
    experimental::CircularBuffer cb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_mask_h = tt::CBIndex::c_5;
    experimental::CircularBuffer cb_mask_h_obj(cb_mask_h);  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;
    experimental::CircularBuffer cb_mask_w_obj(cb_mask_w);  // mask_w

    // Sum[y * dy]
    constexpr auto cb_dgamma = tt::CBIndex::c_16;
    experimental::CircularBuffer cb_dgamma_obj(cb_dgamma);  // gamma_grad(==dgamma)
    // Sum[dy]
    constexpr auto cb_dbeta = tt::CBIndex::c_17;
    experimental::CircularBuffer cb_dbeta_obj(cb_dbeta);  // beta_grad(==dbeta)

    // y = (x - mean) * rstd
    constexpr auto cb_y = tt::CBIndex::c_24;
    experimental::CircularBuffer cb_y_obj(cb_y);  // output(==y)
    constexpr auto cb_ydy = tt::CBIndex::c_25;
    experimental::CircularBuffer cb_ydy_obj(cb_ydy);  // y * dy
    constexpr auto cb_dyadd = tt::CBIndex::c_26;
    experimental::CircularBuffer cb_dyadd_obj(cb_dyadd);  // Add[dy]
    constexpr auto cb_ydyadd = tt::CBIndex::c_27;
    experimental::CircularBuffer cb_ydyadd_obj(cb_ydyadd);  // Add[y * dy]
    constexpr auto cb_xmm = tt::CBIndex::c_28;
    experimental::CircularBuffer cb_xmm_obj(cb_xmm);  // x - mean
    constexpr auto cb_dycopy = tt::CBIndex::c_29;
    experimental::CircularBuffer cb_dycopy_obj(cb_dycopy);  // dycopy

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

    constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, cb_out_init);

    cb_scaler_obj.wait_front(onetile);  // comes from the reader

    if (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
    }
    if (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
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
            cb_dy_obj.wait_front(onetile);  // comes from the reader
            cb_dycopy_obj.reserve_back(onetile);

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

            cb_dy_obj.pop_front(onetile);
            cb_dycopy_obj.push_back(onetile);
            tile_regs_release();

            // Compute cb_dyadd
            cb_dycopy_obj.wait_front(onetile);
            if (beta_grad_has_value) {
                if (inner_idx == 0) {
                    tile_regs_acquire();
                    cb_dyadd_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(cb_dycopy);
                    copy_tile(cb_dycopy, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_dyadd);

                    cb_dyadd_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    cb_dyadd_obj.wait_front(onetile);
                    cb_dyadd_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(cb_dyadd, cb_dycopy);
                    add_tiles(cb_dyadd, cb_dycopy, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_dyadd);

                    cb_dyadd_obj.pop_front(onetile);
                    cb_dyadd_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // beta_grad_has_value
            // We don't pop cb_dycopy here.

            if (gamma_grad_has_value) {
                // Compute cb_xmm
                // x - mean and mask(optional)
                tile_regs_acquire();
                cb_x_obj.wait_front(onetile);     // comes from the reader
                cb_mean_obj.wait_front(onetile);  // comes from the reader
                cb_xmm_obj.reserve_back(onetile);

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

                cb_x_obj.pop_front(onetile);
                cb_mean_obj.pop_front(onetile);
                cb_xmm_obj.push_back(onetile);
                tile_regs_release();

                // Compute cb_y
                // (x - mean) * rstd
                tile_regs_acquire();
                cb_xmm_obj.wait_front(onetile);
                cb_rstd_obj.wait_front(onetile);  // comes from the reader
                cb_y_obj.reserve_back(onetile);

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

                cb_xmm_obj.pop_front(onetile);
                cb_rstd_obj.pop_front(onetile);
                cb_y_obj.push_back(onetile);
                tile_regs_release();

                // Compute cb_ydy
                tile_regs_acquire();
                cb_y_obj.wait_front(onetile);
                cb_ydy_obj.reserve_back(onetile);

                mul_tiles_init_with_dt(cb_y, cb_dycopy);
                mul_tiles(cb_y, cb_dycopy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_ydy);

                cb_y_obj.pop_front(onetile);
                cb_ydy_obj.push_back(onetile);
                tile_regs_release();

                // Compute cb_ydyadd
                if (inner_idx == 0) {
                    tile_regs_acquire();
                    cb_ydy_obj.wait_front(onetile);
                    cb_ydyadd_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(cb_ydy);
                    copy_tile(cb_ydy, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_ydyadd);

                    cb_ydy_obj.pop_front(onetile);
                    cb_ydyadd_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    cb_ydy_obj.wait_front(onetile);
                    cb_ydyadd_obj.wait_front(onetile);
                    cb_ydyadd_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(cb_ydyadd, cb_ydy);
                    add_tiles(cb_ydyadd, cb_ydy, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_ydyadd);

                    cb_ydy_obj.pop_front(onetile);
                    cb_ydyadd_obj.pop_front(onetile);
                    cb_ydyadd_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // gamma_grad_has_value

            cb_dycopy_obj.pop_front(onetile);
        }  // inner_idx loop

        if (gamma_grad_has_value) {
            // Compute cb_dgamma
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_ydyadd, cb_scaler, cb_dgamma, compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                tile_regs_acquire();
                cb_ydyadd_obj.wait_front(onetile);
                cb_dgamma_obj.reserve_back(onetile);

                copy_tile_init_with_dt(cb_ydyadd);
                copy_tile(cb_ydyadd, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dgamma);

                cb_ydyadd_obj.pop_front(onetile);
                cb_dgamma_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // Compute cb_dbeta
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_dyadd, cb_scaler, cb_dbeta, compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                tile_regs_acquire();
                cb_dyadd_obj.wait_front(onetile);
                cb_dbeta_obj.reserve_back(onetile);

                copy_tile_init_with_dt(cb_dyadd);
                copy_tile(cb_dyadd, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dbeta);

                cb_dyadd_obj.pop_front(onetile);
                cb_dbeta_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // beta_grad_has_value

    }  // outer_idx loop
    cb_scaler_obj.pop_front(onetile);

    if (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
    if (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
}
