// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel: bound by moreh_layer_norm_backward's and moreh_group_norm_backward's
// gamma_beta_grad factories. Both bind the same resource names, so a change to this kernel's
// binding vocabulary or argument schema has to land on both factories together.

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto num_cols_per_core = get_arg(args::num_cols_per_core);
    constexpr auto origin_H = get_arg(args::origin_H);
    constexpr auto origin_W = get_arg(args::origin_W);
    constexpr auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool is_lastdim_layernorm = get_arg(args::is_lastdim_layernorm) == 1;
    constexpr bool is_groupnorm = get_arg(args::is_groupnorm) == 1;

    // GAMMA_GRAD_HAS_VALUE / BETA_GRAD_HAS_VALUE / DO_MASK_H / DO_MASK_W arrive as preprocessor
    // defines rather than as arguments, because each one selects whether the host binds a resource:
    // an unbound name does not exist in this build, and even a discarded `if constexpr` branch would
    // still perform name lookup on it. DO_MASK_H / DO_MASK_W were derived here from origin_H /
    // origin_W / is_lastdim_layernorm / is_groupnorm; the host computes the same predicate.

    DataflowBuffer dfb_dy_obj(dfb::dy);          // output_grad(==dy)
    DataflowBuffer dfb_x_obj(dfb::x);            // input(==x)
    DataflowBuffer dfb_mean_obj(dfb::mean);      // mean
    DataflowBuffer dfb_rstd_obj(dfb::rstd);      // rstd
    DataflowBuffer dfb_scaler_obj(dfb::scaler);  // scaler
#ifdef DO_MASK_H
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);  // mask_h
#endif
#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);  // mask_w
#endif

#ifdef GAMMA_GRAD_HAS_VALUE
    // Sum[y * dy]
    DataflowBuffer dfb_dgamma_obj(dfb::dgamma);  // gamma_grad(==dgamma)
#endif
#ifdef BETA_GRAD_HAS_VALUE
    // Sum[dy]
    DataflowBuffer dfb_dbeta_obj(dfb::dbeta);  // beta_grad(==dbeta)
#endif

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

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t Ht = origin_Ht;

    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    constexpr uint32_t HtWt = Ht * Wt;

    // Both operands of this selection resolve at parse time, so the selection itself is gated
    // rather than only its uses: dfb::dgamma does not exist when the host did not bind it.
#ifdef GAMMA_GRAD_HAS_VALUE
    constexpr auto dfb_out_init = dfb::dgamma;
#else
    constexpr auto dfb_out_init = dfb::dbeta;
#endif
    compute_kernel_hw_startup(dfb::dy, dfb::dy, dfb_out_init);

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader

#ifdef DO_MASK_H
    dfb_mask_h_obj.wait_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.wait_front(onetile);
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
            if ((h_idx + 1) % origin_Ht == 0) {
                copy_tile_init_with_dt(dfb_mask_h_obj);
                copy_tile(dfb::mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((w_idx + 1) % origin_Wt == 0) {
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
#ifdef BETA_GRAD_HAS_VALUE
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
#endif  // BETA_GRAD_HAS_VALUE
        // We don't pop dycopy here.

#ifdef GAMMA_GRAD_HAS_VALUE
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
            if ((h_idx + 1) % origin_Ht == 0) {
                copy_tile_init_with_dt(dfb_mask_h_obj);
                copy_tile(dfb::mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((w_idx + 1) % origin_Wt == 0) {
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
#endif  // GAMMA_GRAD_HAS_VALUE

            dfb_dycopy_obj.pop_front(onetile);
        }  // inner_idx loop

#ifdef GAMMA_GRAD_HAS_VALUE
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
#endif  // GAMMA_GRAD_HAS_VALUE

#ifdef BETA_GRAD_HAS_VALUE
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
#endif  // BETA_GRAD_HAS_VALUE

    }  // outer_idx loop
    dfb_scaler_obj.pop_front(onetile);

#ifdef DO_MASK_H
    dfb_mask_h_obj.pop_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.pop_front(onetile);
#endif
}
