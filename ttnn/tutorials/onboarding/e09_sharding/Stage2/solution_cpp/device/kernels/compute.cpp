// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for sharded elementwise add.
// Reads tiles from two sharded input CBs, adds them, writes to sharded output CB.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t tiles_per_shard = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_a = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_b = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    for (uint32_t i = 0; i < tiles_per_shard; i++) {
        tile_regs_acquire();

        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        add_tiles(cb_a, cb_b, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
