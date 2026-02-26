// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Compute kernel for sharded elementwise add.
// Reads tiles from two sharded input CBs, adds them, writes to sharded output CB.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

namespace NAMESPACE {

void MAIN {
    // TODO: Implement compute kernel
    //
    // Compile-time arg: tiles_per_shard at index 0
    //
    // CB indices: cb_a = c_0, cb_b = c_1, cb_out = c_16
    //
    // 1. Call add_tiles_init(cb_a, cb_b)
    // 2. Loop over tiles_per_shard:
    //    a. tile_regs_acquire()
    //    b. cb_wait_front(cb_a, 1); cb_wait_front(cb_b, 1)
    //    c. add_tiles(cb_a, cb_b, 0, 0, 0)
    //    d. tile_regs_commit(); tile_regs_wait()
    //    e. cb_reserve_back(cb_out, 1); pack_tile(0, cb_out); cb_push_back(cb_out, 1)
    //    f. tile_regs_release()
    //    g. cb_pop_front(cb_a, 1); cb_pop_front(cb_b, 1)
}

}  // namespace NAMESPACE
