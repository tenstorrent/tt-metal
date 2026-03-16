// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Compute Kernel (Stage 1: data_pipeline)
// Identity copy: wait for tile in cb_in0, copy to cb_out.
//
// Runtime args:
//   [0] Mt    -- tile rows of C
//   [1] Kt    -- (unused in stage 1)
//   [2] Nt    -- tile columns of C
//   [3] batch -- always 1 for rank-2

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"

constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_out = 16;

namespace NAMESPACE {
void MAIN {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    // Kt unused in stage 1
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);

    // Hardware startup for identity copy (srcA = srcB = cb_in0, dst = cb_out)
    compute_kernel_hw_startup(cb_in0, cb_out);

    copy_tile_to_dst_init_short(cb_in0);

    uint32_t total_tiles = batch * Mt * Nt;
    for (uint32_t i = 0; i < total_tiles; ++i) {
        cb_wait_front(cb_in0, 1);

        acquire_dst();
        copy_tile(cb_in0, 0, 0);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        release_dst();

        cb_pop_front(cb_in0, 1);
    }
}
}  // namespace NAMESPACE
