// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_count = get_compile_time_arg_val(0);
    unary_op_init_common(tt::CB::c_in0);
    for (uint32_t tile_index = 0; tile_index < per_core_tile_count; ++tile_index) {
        cb_wait_front(tt::CB::c_in0, 1);
        cb_reserve_back(tt::CB::c_out0, 1);
        tile_regs_acquire();
        copy_tile(tt::CB::c_in0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CB::c_out0);
        tile_regs_release();
        cb_pop_front(tt::CB::c_in0, 1);
        cb_push_back(tt::CB::c_out0, 1);
    }
}
}  // namespace NAMESPACE
