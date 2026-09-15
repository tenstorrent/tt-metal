// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel: bound by moreh_layer_norm_backward's and moreh_group_norm_backward's
// gamma_beta_grad factories. Both bind the same resource names, so a change to this kernel's
// binding vocabulary or argument schema has to land on both factories together.

#include "moreh_norm_backward_reduce.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto num_cols_per_core = get_arg(args::num_cols_per_core);
    constexpr auto origin_H = get_arg(args::origin_H);
    constexpr auto origin_W = get_arg(args::origin_W);
    constexpr auto NCHt = get_arg(args::NCHt);
    constexpr auto block_tiles = get_arg(args::reduce_block_tiles);
    constexpr auto buffer_tiles = get_arg(args::reduce_buffer_tiles);
#ifdef REDUCE_GRAD_TILES
    constexpr auto num_blocks = NCHt < block_tiles ? 1 : NCHt / block_tiles;
#ifdef BETA_GRAD_HAS_VALUE
    DataflowBuffer reduce_dy(dfb::reduce_dy);
#endif
#ifdef GAMMA_GRAD_HAS_VALUE
    DataflowBuffer reduce_ydy(dfb::reduce_ydy);
#endif
#else
    constexpr uint32_t num_blocks = 1;
#endif
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

    dfb_scaler_obj.wait_front(get_arg(args::reduce_aux_tiles));  // comes from the reader

#ifdef DO_MASK_H
    dfb_mask_h_obj.wait_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.wait_front(onetile);
#endif

    uint32_t h_idx;
    uint32_t w_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_cols_per_core; outer_idx++) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
#ifdef REDUCE_GRAD_TILES
            const uint32_t current_tiles = block + 1 == num_blocks ? NCHt - block * block_tiles : block_tiles;
#ifdef BETA_GRAD_HAS_VALUE
            reduce_dy.reserve_back(buffer_tiles);
#endif
#ifdef GAMMA_GRAD_HAS_VALUE
            reduce_ydy.reserve_back(buffer_tiles);
#endif
#else
            const uint32_t current_tiles = NCHt;
#endif
            for (uint32_t tile = 0; tile < current_tiles; ++tile) {
                const uint32_t inner_idx = block * block_tiles + tile;
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
#ifdef REDUCE_GRAD_TILES
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_dycopy_obj);
            copy_tile(dfb::dycopy, 0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(dfb::reduce_dy);
            pack_tile<true>(dst0, dfb::reduce_dy, tile);
            tile_regs_release();
#else
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
#endif  // REDUCE_GRAD_TILES
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

#ifdef REDUCE_GRAD_TILES
            dfb_ydy_obj.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_ydy_obj);
            copy_tile(dfb::ydy, 0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(dfb::reduce_ydy);
            pack_tile<true>(dst0, dfb::reduce_ydy, tile);
            tile_regs_release();
            dfb_ydy_obj.pop_front(onetile);
#else
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
#endif  // REDUCE_GRAD_TILES
#endif  // GAMMA_GRAD_HAS_VALUE

            dfb_dycopy_obj.pop_front(onetile);
            }  // inner_idx loop

#ifdef REDUCE_GRAD_TILES
#ifdef GAMMA_GRAD_HAS_VALUE
            reduce_ydy.push_back(buffer_tiles);
            reduce_moreh_grad_block<dfb::reduce_ydy, dfb::dgamma, dfb::ydyadd>(block, num_blocks);
            reduce_ydy.pop_front(buffer_tiles);
#endif
#ifdef BETA_GRAD_HAS_VALUE
            reduce_dy.push_back(buffer_tiles);
            reduce_moreh_grad_block<dfb::reduce_dy, dfb::dbeta, dfb::dyadd>(block, num_blocks);
            reduce_dy.pop_front(buffer_tiles);
#endif
#endif
        }  // block loop

#ifndef REDUCE_GRAD_TILES
        // These layer-norm parameters retain every element within a tile;
        // only the outer batch dimension is summed.
#ifdef GAMMA_GRAD_HAS_VALUE
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
#endif
#ifdef BETA_GRAD_HAS_VALUE
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
#endif
#endif

    }  // outer_idx loop
    dfb_scaler_obj.pop_front(get_arg(args::reduce_aux_tiles));

#ifdef DO_MASK_H
    dfb_mask_h_obj.pop_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.pop_front(onetile);
#endif
}
