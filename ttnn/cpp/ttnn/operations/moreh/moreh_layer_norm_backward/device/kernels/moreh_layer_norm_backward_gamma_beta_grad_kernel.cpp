// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

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

    constexpr auto cb_dy = tt::CBIndex::c_0;      // output_grad(==dy)
    constexpr auto cb_x = tt::CBIndex::c_1;       // input(==x)
    constexpr auto cb_mean = tt::CBIndex::c_2;    // mean
    constexpr auto cb_rstd = tt::CBIndex::c_3;    // rstd
    constexpr auto cb_scaler = tt::CBIndex::c_4;  // scaler
    constexpr auto cb_mask_h = tt::CBIndex::c_5;  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;  // mask_w

    // Sum[y * dy]
    constexpr auto cb_dgamma = tt::CBIndex::c_16;  // gamma_grad(==dgamma)
    // Sum[dy]
    constexpr auto cb_dbeta = tt::CBIndex::c_17;  // beta_grad(==dbeta)

    // y = (x - mean) * rstd
    constexpr auto cb_y = tt::CBIndex::c_24;       // output(==y)
    constexpr auto cb_ydy = tt::CBIndex::c_25;     // y * dy
    constexpr auto cb_dyadd = tt::CBIndex::c_26;   // Add[dy]
    constexpr auto cb_ydyadd = tt::CBIndex::c_27;  // Add[y * dy]
    constexpr auto cb_xmm = tt::CBIndex::c_28;     // x - mean
    constexpr auto cb_dycopy = tt::CBIndex::c_29;  // dycopy

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

                // Compute cb_y = (x - mean) * rstd  (T1.24)
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
                                CopyTilePolicy::WaitAndPop,
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
                                CopyTilePolicy::WaitAndPop,
                                CbIndexMode::FirstTile,
                                Dst::D0>{},
                            PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                    }
                }

                // PARTIAL migration: cb_ydy = cb_y * cb_dycopy.
                //   migrated: BinaryFpu(Mul) + PackTile chain.
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
                        CopyTilePolicy::NoWaitNoPop,
                        CbIndexMode::FirstTile,
                        Dst::D0>;
                    eltwise_chain(
                        onetile,
                        MulElt{},
                        PackTile<cb_ydy, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                                 PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
                }

                // Compute cb_ydyadd
                if (inner_idx == 0) {
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
            }  // gamma_grad_has_value

            cb_pop_front(cb_dycopy, onetile);
        }  // inner_idx loop

        if (gamma_grad_has_value) {
            // Compute cb_dgamma
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[y * dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_ydyadd, cb_scaler, cb_dgamma, compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // PARTIAL migration: copy cb_ydyadd -> cb_dgamma.
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_ydyadd);
                pack_reconfig_data_format(cb_dgamma);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_ydyadd, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_dgamma, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // Compute cb_dbeta
            if (is_lastdim_layernorm || is_groupnorm) {
                // Sum[dy]
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_dyadd, cb_scaler, cb_dbeta, compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                // PARTIAL migration: copy cb_dyadd -> cb_dbeta.
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_dyadd);
                pack_reconfig_data_format(cb_dbeta);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_dyadd, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_dbeta, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            }
        }  // beta_grad_has_value

    }  // outer_idx loop
    cb_pop_front(cb_scaler, onetile);

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }
}
