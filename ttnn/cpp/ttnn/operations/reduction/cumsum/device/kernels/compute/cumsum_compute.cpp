// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define APPROX false
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/common.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_3;

    constexpr uint32_t TILE_DEST = 0;
    constexpr uint32_t TILE_ACC = 1;

    constexpr uint32_t first_tile = 0;

    unary_op_init_common(cb_in, cb_out);

    // [UNPACK]: Acquire lock on cb_zero
    // cb_zero is only written once per execution: we can (and should) keep the lock
    cb_wait_front(cb_zero, 1);

    for (unsigned i = 0; i < num_rows; i++) {
        // Initialize cb_intermed
        // Initialize cb_intermed tile (for accumulator)
        // To do this, we fill tile with 0, which are loaded in cb_zero

        // [UNPACK]: Zero => TILE_DEST
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_zero);
        copy_tile(cb_zero, first_tile, TILE_DEST);
        tile_regs_commit();

        // [PACK]: Write TILE_DEST (Zero) to cb_intermed
        tile_regs_wait();
        cb_reserve_back(cb_intermed, 1);
        pack_tile(TILE_DEST, cb_intermed);
        cb_push_back(cb_intermed, 1);
        tile_regs_release();

        for (unsigned j = 0; j < tiles_per_row; j++) {
            // [UNPACK]: Input => TILE_DEST
            cb_wait_front(cb_in, 1);

            // [UNPACK]: Accumulator (db_intermed) => TILE_ACC
            cb_wait_front(cb_intermed, 1);

#ifdef CUMSUM_USE_INT32
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, first_tile, TILE_DEST);

            copy_tile_to_dst_init_short(cb_intermed);
            copy_tile(cb_intermed, first_tile, TILE_ACC);
#endif  // CUMSUM_USE_INT32

            tile_regs_acquire();  // acquire 8 tile registers

#ifndef CUMSUM_USE_INT32
            add_tiles_init(cb_in, cb_intermed);
            add_tiles(cb_in, cb_intermed, 0, 0, TILE_DEST);
#else
            add_int_tile_init();
            add_int32_tile(TILE_DEST, TILE_ACC);
#endif  // CUMSUM_USE_INT32

            tile_regs_commit();

            cb_pop_front(cb_in, 1);
            cb_pop_front(cb_intermed, 1);

            // [PACK]: Write back results
            tile_regs_wait();

            // [PACK]: TILE_DEST => cb_out
            cb_reserve_back(cb_out, 1);
            pack_tile(TILE_DEST, cb_out);
            cb_push_back(cb_out, 1);

            // [PACK]: TILE_DEST => cb_intermed
            cb_reserve_back(cb_intermed, 1);
            pack_tile(TILE_DEST, cb_intermed);
            cb_push_back(cb_intermed, 1);

            tile_regs_release();  // release 8 tile registers
        }

        // If we keep reserve_back() and push_back() into the CircularBuffer
        // then it will eventually get filled if multiple iterations are performed.
        // To avoid this, we pop the circular buffer.
        cb_wait_front(cb_intermed, 1);
        cb_pop_front(cb_intermed, 1);
    }

    cb_pop_front(cb_zero, 1);  // end of kernel: release lock on cb_zero
}

}  // namespace NAMESPACE
