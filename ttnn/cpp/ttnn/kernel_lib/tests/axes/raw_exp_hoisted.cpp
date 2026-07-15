// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RAW-LLK baseline (HOISTED): hand-written exp(x), streaming, per-op init emitted ONCE before the
// loop — the hand-written twin of hoist_single_call.cpp. helper-hoisted must match this (no tax) and
// both must beat raw_exp_unhoisted.cpp (init per tile). exp_tile<false> = exact (Exp<> default). CT: [n].

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

    // Init HOISTED: emitted once for the whole kernel (the chain does this for a uniform chain).
    copy_tile_init(cb_in);
    exp_tile_init<false>();

    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        copy_tile(cb_in, 0, 0);
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
