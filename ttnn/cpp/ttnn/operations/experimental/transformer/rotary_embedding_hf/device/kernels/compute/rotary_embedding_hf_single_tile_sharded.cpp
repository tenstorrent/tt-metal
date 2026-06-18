// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(8);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(9);

    cb_wait_front(trans_mat_cb, onetile);
    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, rotated_in_interm_cb);
    matmul_init(in_cb, trans_mat_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        cb_reserve_back(sin_cb, onetile);
        cb_reserve_back(cos_cb, onetile);
        cb_push_back(sin_cb, onetile);
        cb_push_back(cos_cb, onetile);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            cb_reserve_back(rotated_in_interm_cb, onetile);
            cb_reserve_back(sin_interm_cb, onetile);
            cb_reserve_back(cos_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);

            cb_reserve_back(in_cb, onetile);
            cb_push_back(in_cb, onetile);
            cb_wait_front(in_cb, onetile);

            reconfig_data_format(in_cb, trans_mat_cb);
            pack_reconfig_data_format(rotated_in_interm_cb);
            matmul_init(in_cb, trans_mat_cb);
            tile_regs_acquire();
            matmul_tiles(in_cb, trans_mat_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, rotated_in_interm_cb);
            tile_regs_release();
            cb_push_back(rotated_in_interm_cb, onetile);

            cb_wait_front(rotated_in_interm_cb, onetile);
            cb_wait_front(sin_cb, onetile);
            reconfig_data_format(rotated_in_interm_cb, sin_cb);
            pack_reconfig_data_format(sin_interm_cb);
            tile_regs_acquire();
            mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
            mul_tiles_bcast_rows(rotated_in_interm_cb, sin_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, sin_interm_cb);
            tile_regs_release();
            cb_push_back(sin_interm_cb, onetile);
            cb_pop_front(rotated_in_interm_cb, onetile);

            cb_wait_front(cos_cb, onetile);
            reconfig_data_format(in_cb, cos_cb);
            pack_reconfig_data_format(cos_interm_cb);
            tile_regs_acquire();
            mul_bcast_rows_init_short(in_cb, cos_cb);
            mul_tiles_bcast_rows(in_cb, cos_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cos_interm_cb);
            tile_regs_release();
            cb_push_back(cos_interm_cb, onetile);
            cb_pop_front(in_cb, onetile);

            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            reconfig_data_format(cos_interm_cb, sin_interm_cb);
            pack_reconfig_data_format(out_cb);
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            tile_regs_acquire();
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
            cb_push_back(out_cb, onetile);
            cb_pop_front(cos_interm_cb, onetile);
            cb_pop_front(sin_interm_cb, onetile);
        }

        cb_pop_front(sin_cb, onetile);
        cb_pop_front(cos_cb, onetile);
    }
}
