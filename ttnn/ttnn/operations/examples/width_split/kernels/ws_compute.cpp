// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// width_split example compute (unpack/math/pack TRISCs).
//
// Trivial per-tile op (relu) so the example isolates WORK DISTRIBUTION, not
// compute cost: pull one tile from cb_in, relu it, push to cb_out. Byte-identical
// for both variants; `num_tiles` (this core's share) is the only per-core
// difference. relu is applied once — just enough real math to be an honest op.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_in, cb_out);
    relu_tile_init();

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t t = 0; t < num_tiles; ++t) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);

            tile_regs_acquire();
            copy_tile(cb_in, 0, 0);
            relu_tile(0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_in, 1);
            cb_push_back(cb_out, 1);
        }
    }
}
