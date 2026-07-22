// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t src_cb_id = get_compile_time_arg_val(0);
    uint32_t dst_cb_id = get_compile_time_arg_val(1);
    uint32_t num_tiles = get_compile_time_arg_val(2);
    CircularBuffer src_cb(src_cb_id);
    CircularBuffer dst_cb(dst_cb_id);
    compute_kernel_hw_startup(src_cb_id, dst_cb_id);
    copy_init(src_cb_id);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        src_cb.wait_front(1);
        tile_regs_acquire();
        copy_tile(src_cb_id, 0, 0);
        tile_regs_commit();
        src_cb.pop_front(1);

        dst_cb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dst_cb_id, 0);
        tile_regs_release();
        dst_cb.push_back(1);
    }
}
