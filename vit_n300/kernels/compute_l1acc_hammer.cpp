// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: simple copy from in_cb to out_cb with L1_ACC toggle.
// Tests that pack_reconfig_l1_acc doesn't cause corruption or hang.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    copy_tile_to_dst_init_short(in_cb);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(in_cb, 1);
        tile_regs_acquire();

        copy_tile(in_cb, 0, 0);

        tile_regs_commit();
        cb_pop_front(in_cb, 1);

        cb_reserve_back(out_cb, 1);
        tile_regs_wait();

        // Toggle L1_ACC on every tile - this is the race we want to trigger
        pack_reconfig_l1_acc(t & 1);

        pack_tile(0, out_cb);

        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
}
}  // namespace NAMESPACE
