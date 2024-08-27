// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

constexpr uint32_t onetile = 1;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t num_cols = get_compile_time_arg_val(2);

    constexpr uint32_t cb_src = tt::CB::c_in0;
    constexpr uint32_t cb_zero = tt::CB::c_in1;
    constexpr uint32_t cb_acc = tt::CB::c_intermed0;
    constexpr uint32_t cb_dst = tt::CB::c_out0;

    constexpr uint32_t zero_dst_reg = 0;
    constexpr uint32_t acc_dst_reg = 0;

    binary_op_init_common(cb_src, cb_acc);

    cb_wait_front(cb_zero, onetile);

    for (uint32_t c = 0; c < num_cols; c++) {
        // acc = 0
        cb_reserve_back(cb_acc, onetile);

        tile_regs_acquire();
        copy_tile_init_with_dt(cb_zero);
        copy_tile(cb_zero, 0, zero_dst_reg);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(zero_dst_reg, cb_acc);
        tile_regs_release();

        cb_push_back(cb_acc, onetile);

        cb_wait_front(cb_acc, onetile);

        for (uint32_t n = 0; n < N; n++) {
            // acc = src + acc
            cb_wait_front(cb_src, onetile);

            tile_regs_acquire();
            add_tiles_init();
            add_tiles(cb_src, cb_acc, 0, 0, acc_dst_reg);
            cb_pop_front(cb_acc, onetile);
            cb_reserve_back(cb_acc, onetile);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(acc_dst_reg, cb_acc);
            tile_regs_release();

            cb_push_back(cb_acc, onetile);
            cb_pop_front(cb_src, onetile);

            // dst = acc
            cb_wait_front(cb_acc, onetile);
            cb_reserve_back(cb_dst, onetile);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_acc);
            copy_tile(cb_acc, 0, acc_dst_reg);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(acc_dst_reg, cb_dst);
            tile_regs_release();

            cb_push_back(cb_dst, onetile);
        }

        cb_pop_front(cb_acc, onetile);
    }

    cb_pop_front(cb_zero, onetile);
}
}  // namespace NAMESPACE
