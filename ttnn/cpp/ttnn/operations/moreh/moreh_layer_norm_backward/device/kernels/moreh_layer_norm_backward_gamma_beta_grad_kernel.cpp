// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_cols_per_core = get_arg(args::num_cols_per_core);
    constexpr uint32_t origin_H = get_arg(args::origin_H);
    constexpr uint32_t origin_W = get_arg(args::origin_W);
    constexpr uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr bool gamma_grad_has_value = get_arg(args::gamma_grad_has_value) == 1;
    constexpr bool beta_grad_has_value = get_arg(args::beta_grad_has_value) == 1;
    constexpr bool is_lastdim_layernorm = get_arg(args::is_lastdim_layernorm) == 1;
    constexpr bool is_groupnorm = get_arg(args::is_groupnorm) == 1;

    DataflowBuffer dfb_dy_obj(dfb::dy);          // output_grad(==dy)
    DataflowBuffer dfb_x_obj(dfb::x);            // input(==x)
    DataflowBuffer dfb_mean_obj(dfb::mean);      // mean
    DataflowBuffer dfb_rstd_obj(dfb::rstd);      // rstd
    DataflowBuffer dfb_scaler_obj(dfb::scaler);  // scaler
    // mask_h is allocated only when the last row tile of the normalized region is partial, so its
    // handle exists only on that path — the gate has to run at the preprocessor, before name lookup.
#ifdef DO_MASK_H
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);  // mask_h
#endif
    // mask_w belongs to the groupnorm shape this kernel was written for. The factory hardwires
    // is_groupnorm to false and allocates no mask_w buffer, so nothing binds the handle, DO_MASK_W is
    // never emitted, and the whole path compiles out — exactly as dead as it already was.
    //
    // Kept rather than deleted, deliberately. mask_w is not dead on its own: every is_groupnorm branch
    // in this kernel is unreachable for the same single reason, so dropping mask_w alone would leave
    // the groupnorm scaffolding half torn down while the rest of it still reads as live. Whether to
    // wire groupnorm up or retire it is a behavioral call for the op owner, and either way it is one
    // change covering all of that scaffolding at once.
#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);  // mask_w
#endif

    // Sum[y * dy]
    DataflowBuffer dfb_dgamma_obj(dfb::dgamma);  // gamma_grad(==dgamma)
    // Sum[dy]
    DataflowBuffer dfb_dbeta_obj(dfb::dbeta);  // beta_grad(==dbeta)

    // y = (x - mean) * rstd
    DataflowBuffer dfb_y_obj(dfb::y);            // output(==y)
    DataflowBuffer dfb_ydy_obj(dfb::ydy);        // y * dy
    DataflowBuffer dfb_dyadd_obj(dfb::dyadd);    // Add[dy]
    DataflowBuffer dfb_ydyadd_obj(dfb::ydyadd);  // Add[y * dy]
    DataflowBuffer dfb_xmm_obj(dfb::xmm);        // x - mean
    DataflowBuffer dfb_dycopy_obj(dfb::dycopy);  // dycopy

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

    // Both output buffers are always allocated, so this selection needs no preprocessor gate.
    constexpr auto dfb_out_init = gamma_grad_has_value ? dfb::dgamma : dfb::dbeta;
    compute_kernel_hw_startup(dfb::dy, dfb::dy, dfb_out_init);

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader

#ifdef DO_MASK_H
    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }
#endif
#ifdef DO_MASK_W
    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }
#endif

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

            // Compute dycopy
            // deepcopy and mask(optional)
            tile_regs_acquire();
            dfb_dy_obj.wait_front(onetile);  // comes from the reader
            dfb_dycopy_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_dy_obj);
            copy_tile(dfb::dy, 0, dst0);

#ifdef DO_MASK_H
            if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                copy_tile_init_with_dt(dfb_mask_h_obj);
                copy_tile(dfb::mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(dfb_mask_w_obj);
                copy_tile(dfb::mask_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_dycopy_obj);

            dfb_dy_obj.pop_front(onetile);
            dfb_dycopy_obj.push_back(onetile);
            tile_regs_release();

            // Compute dyadd
            dfb_dycopy_obj.wait_front(onetile);
            if (beta_grad_has_value) {
                if (inner_idx == 0) {
                    tile_regs_acquire();
                    dfb_dyadd_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(dfb_dycopy_obj);
                    copy_tile(dfb::dycopy, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_dyadd_obj);

                    dfb_dyadd_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    dfb_dyadd_obj.wait_front(onetile);
                    dfb_dyadd_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(dfb_dyadd_obj, dfb_dycopy_obj);
                    add_tiles(dfb::dyadd, dfb::dycopy, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_dyadd_obj);

                    dfb_dyadd_obj.pop_front(onetile);
                    dfb_dyadd_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // beta_grad_has_value
            // We don't pop dycopy here.

            if (gamma_grad_has_value) {
                // Compute xmm
                // x - mean and mask(optional)
                tile_regs_acquire();
                dfb_x_obj.wait_front(onetile);     // comes from the reader
                dfb_mean_obj.wait_front(onetile);  // comes from the reader
                dfb_xmm_obj.reserve_back(onetile);

                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_with_dt(dfb_x_obj, dfb_mean_obj);
                    sub_tiles_bcast_cols(dfb::x, dfb::mean, 0, 0, dst0);
                } else {
                    sub_bcast_scalar_init_with_dt(dfb_x_obj, dfb_mean_obj);
                    sub_tiles_bcast_scalar(dfb::x, dfb::mean, 0, 0, dst0);
                }

#ifdef DO_MASK_H
                if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                    copy_tile_init_with_dt(dfb_mask_h_obj);
                    copy_tile(dfb::mask_h, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
#endif

#ifdef DO_MASK_W
                if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(dfb_mask_w_obj);
                    copy_tile(dfb::mask_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
#endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_xmm_obj);

                dfb_x_obj.pop_front(onetile);
                dfb_mean_obj.pop_front(onetile);
                dfb_xmm_obj.push_back(onetile);
                tile_regs_release();

                // Compute y
                // (x - mean) * rstd
                tile_regs_acquire();
                dfb_xmm_obj.wait_front(onetile);
                dfb_rstd_obj.wait_front(onetile);  // comes from the reader
                dfb_y_obj.reserve_back(onetile);

                if (is_lastdim_layernorm) {
                    mul_bcast_cols_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                    mul_tiles_bcast_cols(dfb::xmm, dfb::rstd, 0, 0, dst0);
                } else {
                    mul_bcast_scalar_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                    mul_tiles_bcast_scalar(dfb::xmm, dfb::rstd, 0, 0, dst0);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_y_obj);

                dfb_xmm_obj.pop_front(onetile);
                dfb_rstd_obj.pop_front(onetile);
                dfb_y_obj.push_back(onetile);
                tile_regs_release();

                // Compute ydy
                tile_regs_acquire();
                dfb_y_obj.wait_front(onetile);
                dfb_ydy_obj.reserve_back(onetile);

                mul_tiles_init_with_dt(dfb_y_obj, dfb_dycopy_obj);
                mul_tiles(dfb::y, dfb::dycopy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_ydy_obj);

                dfb_y_obj.pop_front(onetile);
                dfb_ydy_obj.push_back(onetile);
                tile_regs_release();

                // Compute ydyadd
                if (inner_idx == 0) {
                    tile_regs_acquire();
                    dfb_ydy_obj.wait_front(onetile);
                    dfb_ydyadd_obj.reserve_back(onetile);

                    copy_tile_init_with_dt(dfb_ydy_obj);
                    copy_tile(dfb::ydy, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_ydyadd_obj);

                    dfb_ydy_obj.pop_front(onetile);
                    dfb_ydyadd_obj.push_back(onetile);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    dfb_ydy_obj.wait_front(onetile);
                    dfb_ydyadd_obj.wait_front(onetile);
                    dfb_ydyadd_obj.reserve_back(onetile);

                    add_tiles_init_with_dt(dfb_ydyadd_obj, dfb_ydy_obj);
                    add_tiles(dfb::ydyadd, dfb::ydy, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_ydyadd_obj);

                    dfb_ydy_obj.pop_front(onetile);
                    dfb_ydyadd_obj.pop_front(onetile);
                    dfb_ydyadd_obj.push_back(onetile);
                    tile_regs_release();
                }
            }  // gamma_grad_has_value

            dfb_dycopy_obj.pop_front(onetile);
        }  // inner_idx loop

        if (gamma_grad_has_value) {
            // Compute dgamma
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::ydyadd, dfb::scaler, dfb::dgamma>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                tile_regs_acquire();
                dfb_ydyadd_obj.wait_front(onetile);
                dfb_dgamma_obj.reserve_back(onetile);

                copy_tile_init_with_dt(dfb_ydyadd_obj);
                copy_tile(dfb::ydyadd, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_dgamma_obj);

                dfb_ydyadd_obj.pop_front(onetile);
                dfb_dgamma_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // Compute dbeta
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::dyadd, dfb::scaler, dfb::dbeta>(
                    compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // Just copy
                tile_regs_acquire();
                dfb_dyadd_obj.wait_front(onetile);
                dfb_dbeta_obj.reserve_back(onetile);

                copy_tile_init_with_dt(dfb_dyadd_obj);
                copy_tile(dfb::dyadd, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_dbeta_obj);

                dfb_dyadd_obj.pop_front(onetile);
                dfb_dbeta_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // beta_grad_has_value

    }  // outer_idx loop
    dfb_scaler_obj.pop_front(onetile);

#ifdef DO_MASK_H
    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
#endif
#ifdef DO_MASK_W
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
#endif
}
