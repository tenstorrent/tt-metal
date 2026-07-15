// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RAW-LLK baseline (UNHOISTED): like raw_exp_hoisted.cpp but the per-op init is re-emitted every
// iteration (twin of hoist_per_tile.cpp). Only init placement differs, so the gap = per-tile init
// cost — the reference the hoisted side must beat. CT: [n].

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        copy_tile_init(cb_in);  // init re-emitted PER TILE (unhoisted)
        copy_tile(cb_in, 0, 0);
        exp_tile_init<false>();  // init re-emitted PER TILE (unhoisted)
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
