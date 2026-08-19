// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel: bound by moreh_layer_norm_backward's and moreh_group_norm_backward's
// input_grad factories, on the large-algorithm path. Both bind the same resource names, so a change
// to this kernel's binding vocabulary or argument schema has to land on both factories together.

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

void kernel_main() {
    constexpr auto num_rows_per_core = get_arg(args::num_rows_per_core);
    constexpr auto origin_H = get_arg(args::origin_H);
    constexpr auto origin_W = get_arg(args::origin_W);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool is_lastdim_layernorm = get_arg(args::is_lastdim_layernorm) == 1;
    constexpr bool is_groupnorm = get_arg(args::is_groupnorm) == 1;

    // GAMMA_HAS_VALUE / DO_MASK_H / DO_MASK_W arrive as preprocessor defines rather than as
    // arguments, because each selects whether the host binds a resource: an unbound name does not
    // exist in this build, and even a discarded `if constexpr` branch would still look it up.
    // DO_MASK_H / DO_MASK_W were derived here from origin_H / origin_W / is_lastdim_layernorm; the
    // host computes the same predicate.

    compute_kernel_hw_startup(dfb::x, dfb::mean, dfb::dx);

    DataflowBuffer dfb_dy_obj(dfb::dy);                // output_grad(==dy)
    DataflowBuffer dfb_x_obj(dfb::x);                  // input(==x)
    DataflowBuffer dfb_mean_obj(dfb::mean);            // mean
    DataflowBuffer dfb_rstd_obj(dfb::rstd);            // rstd
    DataflowBuffer dfb_scaler_obj(dfb::scaler);        // scaler
    DataflowBuffer dfb_n_recip_n_obj(dfb::n_recip_n);  // n_recip_n
#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma_obj(dfb::gamma);  // gamma
#endif
#if defined(DO_MASK_H) || defined(DO_MASK_W)
    DataflowBuffer dfb_mask_h_w_obj(dfb::mask_h_w);  // mask_h_w
#endif

    // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    DataflowBuffer dfb_dx_obj(dfb::dx);  // input_grad(==dx)

    // y = (x - mean) * rstd
    DataflowBuffer dfb_dycopy_obj(dfb::dycopy);  // copy output_grad(==dycopy)
    DataflowBuffer dfb_y_obj(dfb::y);            // output(==y)
    DataflowBuffer dfb_dysum_obj(dfb::dysum);    // Sum[dy]
    DataflowBuffer dfb_ydysum_obj(dfb::ydysum);  // Sum[y * dy]

    // tmp1..tmp3 are scratch buffers the working names below alias. Each is ONE buffer with one
    // FIFO: the aliases share its read/write pointers, which is what lets a value packed under one
    // name be read back under another. Hence one object per buffer, and references — never a second
    // DataflowBuffer on the same buffer.
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);  // tmp1
    DataflowBuffer dfb_tmp2_obj(dfb::tmp2);  // tmp2
    DataflowBuffer dfb_tmp3_obj(dfb::tmp3);  // tmp3

    constexpr uint32_t onetile = 1;

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader
    dfb_n_recip_n_obj.wait_front(2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    dfb_mask_h_w_obj.wait_front(2);  // comes from the reader
#endif

    constexpr uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        dfb_mean_obj.wait_front(onetile);  // comes from the reader
        dfb_rstd_obj.wait_front(onetile);  // comes from the reader

        // Compute y
        // y = (x - mean) * rstd
        constexpr auto dfb_dyadd = dfb::tmp1;
        auto& dfb_dyadd_obj = dfb_tmp1_obj;
        constexpr auto dfb_ydyadd = dfb::tmp2;
        auto& dfb_ydyadd_obj = dfb_tmp2_obj;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute xmm
            // x - mean
            constexpr auto dfb_xmm = dfb::tmp3;
            auto& dfb_xmm_obj = dfb_tmp3_obj;
            tile_regs_acquire();
            dfb_x_obj.wait_front(onetile);  // comes from the reader
            dfb_xmm_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_with_dt(dfb_x_obj, dfb_mean_obj);
                sub_tiles_bcast_cols(dfb::x, dfb::mean, 0, 0, dst0);
            } else {
                sub_bcast_scalar_init_with_dt(dfb_x_obj, dfb_mean_obj);
                sub_tiles_bcast_scalar(dfb::x, dfb::mean, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_xmm_obj);

            dfb_x_obj.pop_front(onetile);
            dfb_xmm_obj.push_back(onetile);
            tile_regs_release();

            // Compute y
            // (x - mean) * rstd and mask(optional)
            tile_regs_acquire();
            dfb_xmm_obj.wait_front(onetile);
            dfb_y_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                mul_tiles_bcast_cols(dfb_xmm, dfb::rstd, 0, 0, dst0);
            } else {
                mul_bcast_scalar_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                mul_tiles_bcast_scalar(dfb_xmm, dfb::rstd, 0, 0, dst0);
            }

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_y_obj);

            dfb_xmm_obj.pop_front(onetile);
            dfb_y_obj.push_back(onetile);
            tile_regs_release();

            // Copy dy to dycopy
            dfb_dycopy_obj.reserve_back(onetile);
#ifdef GAMMA_HAS_VALUE
            // Compute dycopy
            // dycopy = dy * gamma and mask(optional)
            tile_regs_acquire();
            dfb_dy_obj.wait_front(onetile);     // comes from the reader
            dfb_gamma_obj.wait_front(onetile);  // comes from the reader

            if (is_groupnorm) {
                mul_bcast_scalar_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                mul_tiles_bcast_scalar(dfb::dy, dfb::gamma, 0, 0, dst0);
            } else {
                if (is_lastdim_layernorm) {
                    mul_bcast_rows_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                    mul_tiles_bcast_rows(dfb::dy, dfb::gamma, 0, 0, dst0);
                } else {
                    mul_tiles_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                    mul_tiles(dfb::dy, dfb::gamma, 0, 0, dst0);
                }
            }

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_dycopy_obj);

            dfb_dy_obj.pop_front(onetile);
            dfb_gamma_obj.pop_front(onetile);
            dfb_dycopy_obj.push_back(onetile);
            tile_regs_release();
#else  // GAMMA_HAS_VALUE
       // Compute dycopy
       // dycopy = dy and mask(optional)
            tile_regs_acquire();
            dfb_dy_obj.wait_front(onetile);  // comes from the reader

            copy_tile_init_with_dt(dfb_dy_obj);
            copy_tile(dfb::dy, 0, dst0);

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

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
#endif  // GAMMA_HAS_VALUE

            // Compute dyadd
            dfb_dycopy_obj.wait_front(onetile);
            if (wt == 0) {
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
                add_tiles(dfb_dyadd, dfb::dycopy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_dyadd_obj);

                dfb_dyadd_obj.pop_front(onetile);
                dfb_dyadd_obj.push_back(onetile);
                tile_regs_release();
            }
            // We don't pop dycopy here.

            // Compute ydy and ydyadd
            constexpr auto dfb_ydy = dfb::tmp3;
            auto& dfb_ydy_obj = dfb_tmp3_obj;
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
            dfb_dycopy_obj.pop_front(onetile);
            dfb_ydy_obj.push_back(onetile);
            tile_regs_release();

            // Compute ydyadd
            if (wt == 0) {
                tile_regs_acquire();
                dfb_ydy_obj.wait_front(onetile);
                dfb_ydyadd_obj.reserve_back(onetile);

                copy_tile_init_with_dt(dfb_ydy_obj);
                copy_tile(dfb_ydy, 0, dst0);
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
                add_tiles(dfb_ydyadd, dfb_ydy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_ydyadd_obj);

                dfb_ydy_obj.pop_front(onetile);
                dfb_ydyadd_obj.pop_front(onetile);
                dfb_ydyadd_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // Wt loop

        // Compute dysum
        // Sum[dy]
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb_dyadd, dfb::scaler, dfb::dysum>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        // Compute ydysum
        // Sum[y * dy]
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb_ydyadd, dfb::scaler, dfb::ydysum>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        // Compute recip_nrstd
        // rstd / n -> tmp3
        constexpr auto dfb_recip_nrstd = dfb::tmp3;
        auto& dfb_recip_nrstd_obj = dfb_tmp3_obj;
        tile_regs_acquire();
        dfb_recip_nrstd_obj.reserve_back(onetile);

        if (is_lastdim_layernorm) {
            mul_bcast_cols_init_with_dt(dfb_n_recip_n_obj, dfb_rstd_obj);
            mul_tiles_bcast_cols(dfb::n_recip_n, dfb::rstd, 1, 0, dst0);
        } else {
            mul_bcast_scalar_init_with_dt(dfb_n_recip_n_obj, dfb_rstd_obj);
            mul_tiles_bcast_scalar(dfb::n_recip_n, dfb::rstd, 1, 0, dst0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_recip_nrstd_obj);

        dfb_recip_nrstd_obj.push_back(onetile);
        tile_regs_release();

        // Compute dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        dfb_dysum_obj.wait_front(onetile);
        dfb_ydysum_obj.wait_front(onetile);
        dfb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Copy dy to dycopy
            dfb_dycopy_obj.reserve_back(onetile);
#ifdef GAMMA_HAS_VALUE
            // Compute dycopy
            // dycopy = dy * gamma and mask(optional)
            tile_regs_acquire();
            dfb_dy_obj.wait_front(onetile);     // comes from the reader
            dfb_gamma_obj.wait_front(onetile);  // comes from the reader

            if (is_groupnorm) {
                mul_bcast_scalar_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                mul_tiles_bcast_scalar(dfb::dy, dfb::gamma, 0, 0, dst0);
            } else {
                if (is_lastdim_layernorm) {
                    mul_bcast_rows_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                    mul_tiles_bcast_rows(dfb::dy, dfb::gamma, 0, 0, dst0);
                } else {
                    mul_tiles_init_with_dt(dfb_dy_obj, dfb_gamma_obj);
                    mul_tiles(dfb::dy, dfb::gamma, 0, 0, dst0);
                }
            }

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_dycopy_obj);

            dfb_dy_obj.pop_front(onetile);
            dfb_gamma_obj.pop_front(onetile);
            dfb_dycopy_obj.push_back(onetile);
            tile_regs_release();
#else  // GAMMA_HAS_VALUE
       // Compute dycopy
       // dycopy = dy and mask(optional)
            tile_regs_acquire();
            dfb_dy_obj.wait_front(onetile);  // comes from the reader

            copy_tile_init_with_dt(dfb_dy_obj);
            copy_tile(dfb::dy, 0, dst0);

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

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
#endif  // GAMMA_HAS_VALUE

            // Compute ndy
            // n * dy
            constexpr auto dfb_ndy = dfb::tmp1;
            auto& dfb_ndy_obj = dfb_tmp1_obj;
            tile_regs_acquire();
            dfb_dycopy_obj.wait_front(onetile);
            dfb_ndy_obj.reserve_back(onetile);

            mul_tiles_init_with_dt(dfb_n_recip_n_obj, dfb_dycopy_obj);
            mul_tiles(dfb::n_recip_n, dfb::dycopy, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_ndy_obj);

            dfb_dycopy_obj.pop_front(onetile);
            dfb_ndy_obj.push_back(onetile);
            tile_regs_release();

            // Compute ndymdysum
            // n * dy - Sum[dy]
            constexpr auto dfb_ndymdysum = dfb::tmp2;
            auto& dfb_ndymdysum_obj = dfb_tmp2_obj;
            tile_regs_acquire();
            dfb_ndy_obj.wait_front(onetile);
            dfb_ndymdysum_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_with_dt(dfb_ndy_obj, dfb_dysum_obj);
                sub_tiles_bcast_cols(dfb_ndy, dfb::dysum, 0, 0, dst0);
            } else {
                sub_bcast_scalar_init_with_dt(dfb_ndy_obj, dfb_dysum_obj);
                sub_tiles_bcast_scalar(dfb_ndy, dfb::dysum, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_ndymdysum_obj);

            dfb_ndy_obj.pop_front(onetile);
            dfb_ndymdysum_obj.push_back(onetile);
            tile_regs_release();

            // Compute xmm
            // x - mean and mask(optional)
            constexpr auto dfb_xmm = dfb::tmp1;
            auto& dfb_xmm_obj = dfb_tmp1_obj;
            tile_regs_acquire();
            dfb_x_obj.wait_front(onetile);  // comes from the reader
            dfb_xmm_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_with_dt(dfb_x_obj, dfb_mean_obj);
                sub_tiles_bcast_cols(dfb::x, dfb::mean, 0, 0, dst0);
            } else {
                sub_bcast_scalar_init_with_dt(dfb_x_obj, dfb_mean_obj);
                sub_tiles_bcast_scalar(dfb::x, dfb::mean, 0, 0, dst0);
            }

#ifdef DO_MASK_H
            if (need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif

#ifdef DO_MASK_W
            if ((wt + 1) % origin_Wt == 0) {
                copy_tile_init_with_dt(dfb_mask_h_w_obj);
                copy_tile(dfb::mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_xmm_obj);

            dfb_x_obj.pop_front(onetile);
            dfb_xmm_obj.push_back(onetile);
            tile_regs_release();

            // Compute y
            // (x - mean) * rstd
            tile_regs_acquire();
            dfb_xmm_obj.wait_front(onetile);
            dfb_y_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                mul_tiles_bcast_cols(dfb_xmm, dfb::rstd, 0, 0, dst0);
            } else {
                mul_bcast_scalar_init_with_dt(dfb_xmm_obj, dfb_rstd_obj);
                mul_tiles_bcast_scalar(dfb_xmm, dfb::rstd, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_y_obj);

            dfb_xmm_obj.pop_front(onetile);
            dfb_y_obj.push_back(onetile);
            tile_regs_release();

            // Compute yydysum
            // y * Sum[y * dy]
            constexpr auto dfb_yydysum = dfb::tmp1;
            auto& dfb_yydysum_obj = dfb_tmp1_obj;
            tile_regs_acquire();
            dfb_y_obj.wait_front(onetile);
            dfb_yydysum_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_with_dt(dfb_y_obj, dfb_ydysum_obj);
                mul_tiles_bcast_cols(dfb::y, dfb::ydysum, 0, 0, dst0);
            } else {
                mul_bcast_scalar_init_with_dt(dfb_y_obj, dfb_ydysum_obj);
                mul_tiles_bcast_scalar(dfb::y, dfb::ydysum, 0, 0, dst0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_yydysum_obj);

            dfb_y_obj.pop_front(onetile);
            dfb_yydysum_obj.push_back(onetile);
            tile_regs_release();

            // Compute tmp4
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto dfb_tmp4 = dfb::y;
            auto& dfb_tmp4_obj = dfb_y_obj;
            tile_regs_acquire();
            dfb_ndymdysum_obj.wait_front(onetile);
            dfb_yydysum_obj.wait_front(onetile);
            dfb_tmp4_obj.reserve_back(onetile);

            sub_tiles_init_with_dt(dfb_ndymdysum_obj, dfb_yydysum_obj);
            sub_tiles(dfb_ndymdysum, dfb_yydysum, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_tmp4_obj);

            dfb_ndymdysum_obj.pop_front(onetile);
            dfb_yydysum_obj.pop_front(onetile);
            dfb_tmp4_obj.push_back(onetile);
            tile_regs_release();

            // Compute dx
            // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
            tile_regs_acquire();
            dfb_tmp4_obj.wait_front(onetile);
            dfb_dx_obj.reserve_back(onetile);

            mul_tiles_init_with_dt(dfb_tmp4_obj, dfb_recip_nrstd_obj);
            mul_tiles(dfb_tmp4, dfb_recip_nrstd, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_dx_obj);

            dfb_tmp4_obj.pop_front(onetile);
            dfb_dx_obj.push_back(onetile);
            tile_regs_release();
        }  // Wt loop
        dfb_recip_nrstd_obj.pop_front(onetile);
        dfb_dysum_obj.pop_front(onetile);
        dfb_ydysum_obj.pop_front(onetile);

        dfb_mean_obj.pop_front(onetile);
        dfb_rstd_obj.pop_front(onetile);
    }  // NCHt loop
    dfb_scaler_obj.pop_front(onetile);
    dfb_n_recip_n_obj.pop_front(2);

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    dfb_mask_h_w_obj.pop_front(2);
#endif
}
