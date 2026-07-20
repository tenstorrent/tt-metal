// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // add/sub/mul

namespace ckl = compute_kernel_lib;

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

void kernel_main() {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool gamma_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(5) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(6) == 1;

    binary_op_init_common(tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_16);

    constexpr auto cb_dy = tt::CBIndex::c_0;
    CircularBuffer cb_dy_obj(cb_dy);  // output_grad(==dy)
    constexpr auto cb_x = tt::CBIndex::c_1;
    CircularBuffer cb_x_obj(cb_x);  // input(==x)
    constexpr auto cb_mean = tt::CBIndex::c_2;
    CircularBuffer cb_mean_obj(cb_mean);  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_3;
    CircularBuffer cb_rstd_obj(cb_rstd);  // rstd
    constexpr auto cb_scaler = tt::CBIndex::c_4;
    CircularBuffer cb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_n_recip_n = tt::CBIndex::c_5;
    CircularBuffer cb_n_recip_n_obj(cb_n_recip_n);  // n_recip_n
    constexpr auto cb_gamma = tt::CBIndex::c_6;
    CircularBuffer cb_gamma_obj(cb_gamma);  // gamma
    constexpr auto cb_mask_h_w = tt::CBIndex::c_7;
    CircularBuffer cb_mask_h_w_obj(cb_mask_h_w);  // mask_h_w

    // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    constexpr auto cb_dx = tt::CBIndex::c_16;
    CircularBuffer cb_dx_obj(cb_dx);  // input_grad(==dx)

    // y = (x - mean) * rstd
    constexpr auto cb_dycopy = tt::CBIndex::c_24;
    CircularBuffer cb_dycopy_obj(cb_dycopy);  // copy output_grad(==dycopy)
    constexpr auto cb_y = tt::CBIndex::c_25;
    CircularBuffer cb_y_obj(cb_y);  // output(==y)
    constexpr auto cb_dysum = tt::CBIndex::c_26;
    CircularBuffer cb_dysum_obj(cb_dysum);  // Sum[dy]
    constexpr auto cb_ydysum = tt::CBIndex::c_27;
    CircularBuffer cb_ydysum_obj(cb_ydysum);  // Sum[y * dy]

    constexpr auto cb_tmp1 = tt::CBIndex::c_28;  // tmp1
    constexpr auto cb_tmp2 = tt::CBIndex::c_29;  // tmp2
    constexpr auto cb_tmp3 = tt::CBIndex::c_30;  // tmp3

    constexpr uint32_t onetile = 1;

    cb_scaler_obj.wait_front(onetile);  // comes from the reader
    cb_n_recip_n_obj.wait_front(2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    if (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    constexpr uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        cb_mean_obj.wait_front(onetile);  // comes from the reader
        cb_rstd_obj.wait_front(onetile);  // comes from the reader

        // Compute cb_y
        // y = (x - mean) * rstd
        constexpr auto cb_dyadd = cb_tmp1;
        CircularBuffer cb_dyadd_obj(cb_dyadd);
        constexpr auto cb_ydyadd = cb_tmp2;
        CircularBuffer cb_ydyadd_obj(cb_ydyadd);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_xmm
            // x - mean
            constexpr auto cb_xmm = cb_tmp3;
            CircularBuffer cb_xmm_obj(cb_xmm);
            ckl::sub<
                cb_x,
                cb_mean,
                cb_xmm,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_y
            // (x - mean) * rstd and mask(optional)
            tile_regs_acquire();
            cb_xmm_obj.wait_front(onetile);
            cb_y_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_short_with_dt(DataflowBuffer(cb_xmm), DataflowBuffer(cb_rstd));
                mul_tiles_bcast_cols(cb_xmm, cb_rstd, 0, 0, dst0);
            } else {
                mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer(cb_xmm), DataflowBuffer(cb_rstd));
                mul_tiles_bcast_scalar(cb_xmm, cb_rstd, 0, 0, dst0);
            }

            if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                copy_tile(cb_mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                copy_tile(cb_mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, DataflowBuffer(cb_y));

            cb_xmm_obj.pop_front(onetile);
            cb_y_obj.push_back(onetile);
            tile_regs_release();

            // Copy cb_dy to cb_dycopy
            cb_dycopy_obj.reserve_back(onetile);
            if (gamma_has_value) {
                // Compute cb_dycopy
                // dycopy = dy * gamma and mask(optional)
                tile_regs_acquire();
                cb_dy_obj.wait_front(onetile);     // comes from the reader
                cb_gamma_obj.wait_front(onetile);  // comes from the reader

                if (is_groupnorm) {
                    mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                    mul_tiles_bcast_scalar(cb_dy, cb_gamma, 0, 0, dst0);
                } else {
                    if (is_lastdim_layernorm) {
                        mul_bcast_rows_init_short_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                        mul_tiles_bcast_rows(cb_dy, cb_gamma, 0, 0, dst0);
                    } else {
                        mul_tiles_init_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                        mul_tiles(cb_dy, cb_gamma, 0, 0, dst0);
                    }
                }

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dycopy));

                cb_dy_obj.pop_front(onetile);
                cb_gamma_obj.pop_front(onetile);
                cb_dycopy_obj.push_back(onetile);
                tile_regs_release();
            } else {
                // Compute cb_dycopy
                // dycopy = dy and mask(optional)
                tile_regs_acquire();
                cb_dy_obj.wait_front(onetile);  // comes from the reader

                copy_tile_init_with_dt(DataflowBuffer(cb_dy));
                copy_tile(cb_dy, 0, dst0);

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dycopy));

                cb_dy_obj.pop_front(onetile);
                cb_dycopy_obj.push_back(onetile);
                tile_regs_release();
            }

            // Compute cb_dyadd
            cb_dycopy_obj.wait_front(onetile);
            if (wt == 0) {
                tile_regs_acquire();
                cb_dyadd_obj.reserve_back(onetile);

                copy_tile_init_with_dt(DataflowBuffer(cb_dycopy));
                copy_tile(cb_dycopy, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dyadd));

                cb_dyadd_obj.push_back(onetile);
                tile_regs_release();
            } else {
                tile_regs_acquire();
                cb_dyadd_obj.wait_front(onetile);
                cb_dyadd_obj.reserve_back(onetile);

                add_tiles_init_with_dt(DataflowBuffer(cb_dyadd), DataflowBuffer(cb_dycopy));
                add_tiles(cb_dyadd, cb_dycopy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dyadd));

                cb_dyadd_obj.pop_front(onetile);
                cb_dyadd_obj.push_back(onetile);
                tile_regs_release();
            }
            // We don't pop cb_dycopy here.

            // Compute cb_ydy and cb_ydyadd
            constexpr auto cb_ydy = cb_tmp3;
            CircularBuffer cb_ydy_obj(cb_ydy);
            ckl::mul<
                cb_y,
                cb_dycopy,
                cb_ydy,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::NoWaitPop)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_ydyadd
            if (wt == 0) {
                tile_regs_acquire();
                cb_ydy_obj.wait_front(onetile);
                cb_ydyadd_obj.reserve_back(onetile);

                copy_tile_init_with_dt(DataflowBuffer(cb_ydy));
                copy_tile(cb_ydy, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_ydyadd));

                cb_ydy_obj.pop_front(onetile);
                cb_ydyadd_obj.push_back(onetile);
                tile_regs_release();
            } else {
                tile_regs_acquire();
                cb_ydy_obj.wait_front(onetile);
                cb_ydyadd_obj.wait_front(onetile);
                cb_ydyadd_obj.reserve_back(onetile);

                add_tiles_init_with_dt(DataflowBuffer(cb_ydyadd), DataflowBuffer(cb_ydy));
                add_tiles(cb_ydyadd, cb_ydy, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_ydyadd));

                cb_ydy_obj.pop_front(onetile);
                cb_ydyadd_obj.pop_front(onetile);
                cb_ydyadd_obj.push_back(onetile);
                tile_regs_release();
            }
        }  // Wt loop

        // Compute cb_dysum
        // Sum[dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_dyadd, cb_scaler, cb_dysum>(ckl::ReduceInputBlockShape::single());

        // Compute cb_ydysum
        // Sum[y * dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_ydyadd, cb_scaler, cb_ydysum>(ckl::ReduceInputBlockShape::single());

        // Compute cb_recip_nrstd
        // rstd / n -> cb_tmp3
        constexpr auto cb_recip_nrstd = cb_tmp3;
        CircularBuffer cb_recip_nrstd_obj(cb_recip_nrstd);
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_n_recip_n,
                cb_rstd,
                ckl::BinaryFpuOp::Mul,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{1u, 0u},
            ckl::PackTile<cb_recip_nrstd>{});

        // Compute cb_dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        cb_dysum_obj.wait_front(onetile);
        cb_ydysum_obj.wait_front(onetile);
        cb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Copy cb_dy to cb_dycopy
            cb_dycopy_obj.reserve_back(onetile);
            if (gamma_has_value) {
                // Compute cb_dycopy
                // dycopy = dy * gamma and mask(optional)
                tile_regs_acquire();
                cb_dy_obj.wait_front(onetile);     // comes from the reader
                cb_gamma_obj.wait_front(onetile);  // comes from the reader

                if (is_groupnorm) {
                    mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                    mul_tiles_bcast_scalar(cb_dy, cb_gamma, 0, 0, dst0);
                } else {
                    if (is_lastdim_layernorm) {
                        mul_bcast_rows_init_short_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                        mul_tiles_bcast_rows(cb_dy, cb_gamma, 0, 0, dst0);
                    } else {
                        mul_tiles_init_with_dt(DataflowBuffer(cb_dy), DataflowBuffer(cb_gamma));
                        mul_tiles(cb_dy, cb_gamma, 0, 0, dst0);
                    }
                }

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dycopy));

                cb_dy_obj.pop_front(onetile);
                cb_gamma_obj.pop_front(onetile);
                cb_dycopy_obj.push_back(onetile);
                tile_regs_release();
            } else {
                // Compute cb_dycopy
                // dycopy = dy and mask(optional)
                tile_regs_acquire();
                cb_dy_obj.wait_front(onetile);  // comes from the reader

                copy_tile_init_with_dt(DataflowBuffer(cb_dy));
                copy_tile(cb_dy, 0, dst0);

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, DataflowBuffer(cb_dycopy));

                cb_dy_obj.pop_front(onetile);
                cb_dycopy_obj.push_back(onetile);
                tile_regs_release();
            }

            // Compute cb_ndy
            // n * dy
            constexpr auto cb_ndy = cb_tmp1;
            CircularBuffer cb_ndy_obj(cb_ndy);
            ckl::mul<
                cb_n_recip_n,
                cb_dycopy,
                cb_ndy,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_ndymdysum
            // n * dy - Sum[dy]
            constexpr auto cb_ndymdysum = cb_tmp2;
            CircularBuffer cb_ndymdysum_obj(cb_ndymdysum);
            ckl::sub<
                cb_ndy,
                cb_dysum,
                cb_ndymdysum,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_xmm
            // x - mean and mask(optional)
            constexpr auto cb_xmm = cb_tmp1;
            CircularBuffer cb_xmm_obj(cb_xmm);
            tile_regs_acquire();
            cb_x_obj.wait_front(onetile);  // comes from the reader
            cb_xmm_obj.reserve_back(onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short_with_dt(DataflowBuffer(cb_x), DataflowBuffer(cb_mean));
                sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer(cb_x), DataflowBuffer(cb_mean));
                sub_tiles_bcast_scalar(cb_x, cb_mean, 0, 0, dst0);
            }

            if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                copy_tile(cb_mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                copy_tile_init_with_dt(DataflowBuffer(cb_mask_h_w));
                copy_tile(cb_mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, DataflowBuffer(cb_xmm));

            cb_x_obj.pop_front(onetile);
            cb_xmm_obj.push_back(onetile);
            tile_regs_release();

            // Compute cb_y
            ckl::mul<
                cb_xmm,
                cb_rstd,
                cb_y,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_yydysum
            // y * Sum[y * dy]
            constexpr auto cb_yydysum = cb_tmp1;
            CircularBuffer cb_yydysum_obj(cb_yydysum);
            ckl::mul<
                cb_y,
                cb_ydysum,
                cb_yydysum,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_tmp4
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto cb_tmp4 = cb_y;
            CircularBuffer cb_tmp4_obj(cb_tmp4);
            ckl::sub<cb_ndymdysum, cb_yydysum, cb_tmp4>(ckl::EltwiseShape::tiles(onetile));

            // Compute cb_dx
            ckl::mul<
                cb_tmp4,
                cb_recip_nrstd,
                cb_dx,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));
        }  // Wt loop
        cb_recip_nrstd_obj.pop_front(onetile);
        cb_dysum_obj.pop_front(onetile);
        cb_ydysum_obj.pop_front(onetile);

        cb_mean_obj.pop_front(onetile);
        cb_rstd_obj.pop_front(onetile);
    }  // NCHt loop
    cb_scaler_obj.pop_front(onetile);
    cb_n_recip_n_obj.pop_front(2);

    if (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.pop_front(2);
    }
}
