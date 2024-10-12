// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t src_cb_id = get_compile_time_arg_val(0);
    uint32_t dst_cb_id = get_compile_time_arg_val(1);
    uint32_t num_tiles = get_compile_time_arg_val(2);
    unary_op_init_common(src_cb_id, dst_cb_id);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(src_cb_id, 1);
        tile_regs_acquire();
        copy_tile(src_cb_id, 0, 0);
        tile_regs_commit();
        cb_pop_front(src_cb_id, 1);

        cb_reserve_back(dst_cb_id, 1);
        tile_regs_wait();
        pack_tile(0, dst_cb_id, 0);
        tile_regs_release();
        cb_push_back(dst_cb_id, 1);
    }
}
}  // namespace NAMESPACE
