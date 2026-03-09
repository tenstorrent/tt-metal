// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Circular buffer indices
    constexpr auto cb_in0 = tt::CBIndex::c_0;       // Input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;       // Input B
    constexpr auto cb_scalar = tt::CBIndex::c_2;    // Scalar for broadcast multiply
    constexpr auto cb_out0 = tt::CBIndex::c_16;     // Output

    // Initialize binary operations
    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    // Wait for scalar and all input tiles to be ready
    cb_wait_front(cb_scalar, 1);
    cb_wait_front(cb_in0, n_tiles);
    cb_wait_front(cb_in1, n_tiles);

    // Reserve space for all output tiles
    cb_reserve_back(cb_out0, n_tiles);

    // Process each tile
    for (uint32_t i = 0; i < n_tiles; i++) {
        tile_regs_acquire();

        // Step 1: B[i] * scalar -> DST[0]
        mul_tiles_bcast_scalar_init_short(cb_in1, cb_scalar);
        mul_tiles_bcast<BroadcastType::SCALAR>(cb_in1, cb_scalar, i, 0, 0);

        // Step 2: A[i] + DST[0] -> DST[0]
        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in0);
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in0, i, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Pack result to output position i
        pack_tile(0, cb_out0, i);

        tile_regs_release();
    }

    // Push all output tiles
    cb_push_back(cb_out0, n_tiles);

    // Pop input tiles
    cb_pop_front(cb_in0, n_tiles);
    cb_pop_front(cb_in1, n_tiles);
}
