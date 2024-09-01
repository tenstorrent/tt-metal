// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/cumsum.h"
#include "compute_kernel_api/transpose_wh.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

constexpr uint32_t onetile = 1;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr bool flip = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t mask_w = get_compile_time_arg_val(3);

    constexpr uint32_t cb_src = tt::CB::c_in0;
    constexpr uint32_t cb_acc = tt::CB::c_intermed0;
    constexpr uint32_t cb_trp = tt::CB::c_intermed1;
    constexpr uint32_t cb_dst = tt::CB::c_out0;

    constexpr uint32_t data_dst_reg = 0;
    constexpr uint32_t acc_dst_reg = data_dst_reg + 1;

    unary_op_init_common(cb_src, cb_dst);

    cb_reserve_back(cb_acc, onetile);
    cb_push_back(cb_acc, onetile);

    for (uint32_t r = 0; r < num_rows; r++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(cb_src, onetile);
            cb_wait_front(cb_acc, onetile);
            cb_reserve_back(cb_trp, onetile);

            tile_regs_acquire();
#ifdef FP32_DEST_ACC_EN
            unpack_reconfig_data_format_srca(cb_src);
#endif
            transpose_wh_init_short(cb_src);
            transpose_wh_tile(cb_src, 0, data_dst_reg);
            copy_tile_init_with_dt(cb_acc);
            copy_tile(cb_acc, 0, acc_dst_reg);
#ifdef DATA_FLOAT
            if (flip) {
                cumsum_row_flip_tile_init();
                cumsum_row_flip_tile(data_dst_reg, acc_dst_reg, wt == 0 ? mask_w : TILE_WIDTH, wt == 0, wt == Wt - 1);
            } else {
                cumsum_row_tile_init();
                cumsum_row_tile(data_dst_reg, acc_dst_reg, wt == 0, wt == Wt - 1);
            }
#else
            cumsum_row_int_tile_init();
            cumsum_row_int_tile(data_dst_reg, acc_dst_reg, wt == 0, wt == Wt - 1);
#endif
            tile_regs_commit();

            cb_pop_front(cb_acc, onetile);
            cb_reserve_back(cb_acc, onetile);

            tile_regs_wait();
            pack_tile_with_dt(data_dst_reg, cb_trp);
            pack_tile(acc_dst_reg, cb_acc);
            tile_regs_release();

            cb_push_back(cb_trp, onetile);
            cb_push_back(cb_acc, onetile);
            cb_pop_front(cb_src, onetile);

            cb_wait_front(cb_trp, onetile);
            cb_reserve_back(cb_dst, onetile);

            tile_regs_acquire();
#ifdef FP32_DEST_ACC_EN
            unpack_reconfig_data_format_srca(cb_trp);
#endif
            transpose_wh_init_short(cb_trp);
            transpose_wh_tile(cb_trp, 0, data_dst_reg);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(data_dst_reg, cb_dst);
            tile_regs_release();

            cb_push_back(cb_dst, onetile);
            cb_pop_front(cb_trp, onetile);
        }
    }
}
}  // namespace NAMESPACE
