// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/sfpu_int_sum.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_mask_w = tt::CB::c_in1;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr int onetile = 1;
    constexpr int idx0 = 0;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    unary_op_init_common(cb_in0, cb_out0);

    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        constexpr bool is_single_wt = (Wt == 1);
        if (is_single_wt) {
            tile_regs_acquire();
            copy_tile_to_dst(cb_in0, idx0, dst0);

            if (do_mask_w) {
                copy_tile_to_dst(cb_mask_w, idx0, dst1, false);
                mask_tile_init();
                mask_tile(dst0, dst1, DataFormat::Int32);
            }

            sfpu_sum_int_init();
            sfpu_sum_int_row(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_from_dst(cb_out0, dst0);
            tile_regs_release();
        } else {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                if (wt == 0) {
                    tile_regs_acquire();
                    copy_tile_to_dst(cb_in0, idx0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_from_dst(cb_intermed0, dst0);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    copy_tile_to_dst(cb_in0, idx0, dst0);
                    if (wt == Wt - 1 && do_mask_w) {
                        copy_tile_to_dst(cb_mask_w, idx0, dst1, false);
                        mask_tile_init();
                        mask_tile(dst0, dst1, DataFormat::Int32);
                    }

                    copy_tile_to_dst(cb_intermed0, idx0, dst1);
                    sfpu_sum_int_init();
                    sfpu_add_int(dst0, dst1);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_from_dst(cb_intermed0, dst0);
                    tile_regs_release();
                }
            }

            tile_regs_acquire();
            copy_tile_to_dst(cb_intermed0, idx0, dst0);
            sfpu_sum_int_init();
            sfpu_sum_int_row(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_from_dst(cb_out0, dst0);
            tile_regs_release();
        }
    }
}
}  // namespace NAMESPACE
