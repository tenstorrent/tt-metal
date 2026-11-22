// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A worker compute: tilize my slice; + se's tiles.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), SE = get_compile_time_arg_val(1);
    const uint32_t RT = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 2, 3);
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t tcb = SE ? 1 : 3;
        cb_wait_front(0, W);
        cb_reserve_back(tcb, W);
        reconfig_data_format_srca(0);
        pack_reconfig_data_format(tcb);
        tilize_init(0, W, tcb);
        tilize_block(0, W, tcb);
        tilize_uninit(0, tcb);
        cb_push_back(tcb, W);
        cb_pop_front(0, W);
        if constexpr (SE) {
            cb_wait_front(1, W);
            cb_wait_front(2, W);
            cb_reserve_back(3, W);
            reconfig_data_format(1, 2);
            pack_reconfig_data_format(3);
            add_init(1, 2);
            tile_regs_acquire();
            for (uint32_t t = 0; t < W; ++t) {
                add_tiles(1, 2, t, t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < W; ++t) {
                pack_tile(t, 3);
            }
            tile_regs_release();
            cb_push_back(3, W);
            cb_pop_front(1, W);
            cb_pop_front(2, W);
        }
    }
}
