// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

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
            // Compute cb_xmm = x - mean (bcast, no mask)  (T1.18)
            constexpr auto cb_xmm = cb_tmp3;
            {
                using namespace compute_kernel_lib;
                if constexpr (is_lastdim_layernorm) {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_x,
                            cb_mean,
                            cb_xmm,
                            BinaryFpuOp::Sub,
                            BroadcastDim::Col,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_xmm, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                } else {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_x,
                            cb_mean,
                            cb_xmm,
                            BinaryFpuOp::Sub,
                            BroadcastDim::Scalar,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_xmm, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }

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
                // PARTIAL migration: seed cb_dyadd with first cb_dycopy tile (no pop on cb_dycopy).
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_dycopy);
                pack_reconfig_data_format(cb_dyadd);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_dycopy, Dst::D0, CopyTilePolicy::WaitNoPop>{},
                        PackTile<cb_dyadd, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
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

            // PARTIAL migration: cb_ydy = cb_y * cb_dycopy (single tile, both WaitAndPop).
            //   migrated: BinaryFpu(Mul) + PackTile chain.
            constexpr auto cb_ydy = cb_tmp3;
            {
                using namespace compute_kernel_lib;
                using MulElt = BinaryFpu<
                    cb_y,
                    cb_dycopy,
                    /*CbOut=*/0,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>;
                eltwise_chain(
                    onetile,
                    MulElt{},
                    PackTile<cb_ydy, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                             PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
            }

            // Compute cb_ydyadd
            if (wt == 0) {
                // PARTIAL migration: seed cb_ydyadd with first cb_ydy tile.
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_ydy);
                pack_reconfig_data_format(cb_ydyadd);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_ydy, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_ydyadd, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
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
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_dyadd, cb_scaler, cb_dysum, compute_kernel_lib::ReduceInputBlockShape::single());

        // Compute cb_ydysum
        // Sum[y * dy]
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_ydyadd, cb_scaler, cb_ydysum, compute_kernel_lib::ReduceInputBlockShape::single());

        // PARTIAL migration: cb_recip_nrstd = cb_n_recip_n[1] * cb_rstd[0] (bcast).
        //   migrated: BinaryFpu(Mul, Cols|Scalar) + PackTile chain.
        constexpr auto cb_recip_nrstd = cb_tmp3;
        {
            using namespace compute_kernel_lib;
            constexpr BroadcastDim BCAST_DIM = is_lastdim_layernorm ? BroadcastDim::Col : BroadcastDim::Scalar;
            using MulBcast = BinaryFpu<
                cb_n_recip_n,
                cb_rstd,
                cb_recip_nrstd,
                BinaryFpuOp::Mul,
                BCAST_DIM,
                BinaryDataFormatReconfig::Input,
                CopyTilePolicy::NoWaitNoPop,
                CopyTilePolicy::NoWaitNoPop,
                CbIndexMode::Pinned,
                Dst::D0>;
            eltwise_chain(
                onetile,
                MulBcast{/*a=*/1, /*b=*/0},
                PackTile<
                    cb_recip_nrstd,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

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

            // PARTIAL migration: cb_ndy = cb_n_recip_n[0] * cb_dycopy.
            //   migrated: BinaryFpu(Mul) + PackTile chain.
            constexpr auto cb_ndy = cb_tmp1;
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_n_recip_n, cb_dycopy);
            pack_reconfig_data_format(cb_ndy);
#endif
            {
                using namespace compute_kernel_lib;
                using MulElt = BinaryFpu<
                    cb_n_recip_n,
                    cb_dycopy,
                    /*CbOut=*/0,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::None,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>;
                eltwise_chain(
                    onetile,
                    MulElt{},
                    PackTile<cb_ndy, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            }

            // Compute cb_ndymdysum = n * dy - Sum[dy] (bcast)  (T1.19)
            constexpr auto cb_ndymdysum = cb_tmp2;
            {
                using namespace compute_kernel_lib;
                if constexpr (is_lastdim_layernorm) {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_ndy,
                            cb_dysum,
                            cb_ndymdysum,
                            BinaryFpuOp::Sub,
                            BroadcastDim::Col,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_ndymdysum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                } else {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_ndy,
                            cb_dysum,
                            cb_ndymdysum,
                            BinaryFpuOp::Sub,
                            BroadcastDim::Scalar,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_ndymdysum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }

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

            // Compute cb_y = (x - mean) * rstd  (T1.20, no-mask path)
            {
                using namespace compute_kernel_lib;
                if constexpr (is_lastdim_layernorm) {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_xmm,
                            cb_rstd,
                            cb_y,
                            BinaryFpuOp::Mul,
                            BroadcastDim::Col,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                } else {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_xmm,
                            cb_rstd,
                            cb_y,
                            BinaryFpuOp::Mul,
                            BroadcastDim::Scalar,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }

            // Compute cb_yydysum = y * Sum[y * dy] (bcast)  (T1.21, both indices 0)
            constexpr auto cb_yydysum = cb_tmp1;
            {
                using namespace compute_kernel_lib;
                if constexpr (is_lastdim_layernorm) {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_y,
                            cb_ydysum,
                            cb_yydysum,
                            BinaryFpuOp::Mul,
                            BroadcastDim::Col,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_yydysum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                } else {
                    eltwise_chain(
                        onetile,
                        BinaryFpu<
                            cb_y,
                            cb_ydysum,
                            cb_yydysum,
                            BinaryFpuOp::Mul,
                            BroadcastDim::Scalar,
                            BinaryDataFormatReconfig::InputAndOutput,
                            CopyTilePolicy::WaitAndPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::FirstTile,
                            Dst::D0>{},
                        PackTile<cb_yydysum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }

            // PARTIAL migration: cb_tmp4 = cb_ndymdysum - cb_yydysum.
            //   migrated: BinaryFpu(Sub) + PackTile chain.
            constexpr auto cb_tmp4 = cb_y;
            {
                using namespace compute_kernel_lib;
                using SubElt = BinaryFpu<
                    cb_ndymdysum,
                    cb_yydysum,
                    /*CbOut=*/0,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>;
                eltwise_chain(
                    onetile,
                    SubElt{},
                    PackTile<cb_tmp4, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                             PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
            }

            // PARTIAL migration: cb_dx = cb_tmp4 * cb_recip_nrstd.
            //   migrated: BinaryFpu(Mul) + PackTile chain.
            {
                using namespace compute_kernel_lib;
                using MulElt = BinaryFpu<
                    cb_tmp4,
                    cb_recip_nrstd,
                    /*CbOut=*/0,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>;
                eltwise_chain(
                    onetile,
                    MulElt{},
                    PackTile<cb_dx, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                             PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
            }
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
}
