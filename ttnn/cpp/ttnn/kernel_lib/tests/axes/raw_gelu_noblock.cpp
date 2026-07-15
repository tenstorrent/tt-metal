// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RAW-LLK baseline for block_size: hand-written gelu(A)*B, block_size=1 (one tile/acquire), Bulk,
// gelu LUT init re-loaded PER TILE. helper blk=1 must match this (no tax); helper blk>1 amortizes the
// LUT that the raw pays every tile. Golden: gelu(A)*B. CT args: [n].

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/gelu.h"   // gelu_tile / gelu_tile_init
#include "api/compute/eltwise_binary_sfpu.h"  // mul_binary_tile / mul_binary_tile_init

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_out);

    // BULK: stage both inputs upfront, drain the output at end (same as the helper).
    cb_wait_front(cb_a, n);
    cb_wait_front(cb_b, n);
    cb_reserve_back(cb_out, n);

    // NO blocking: the expensive gelu LUT init is re-loaded for every tile.
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile_init(cb_a);
        copy_tile(cb_a, i, 0);  // D0 = A[i]
        copy_tile_init(cb_b);
        copy_tile(cb_b, i, 1);  // D1 = B[i]
        gelu_tile_init();       // <-- expensive SFPU LUT load, PER TILE
        gelu_tile(0);           // D0 = gelu(A[i])
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // D0 = gelu(A[i]) * B[i]  (clobbers the gelu LUT)
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);  // sequential: advances the reserved-window write ptr 0..n-1
        tile_regs_release();
    }

    cb_pop_front(cb_a, n);
    cb_pop_front(cb_b, n);
    cb_push_back(cb_out, n);
}
