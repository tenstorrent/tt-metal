// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Sum `ring_size` 1x32 RM faces that already sit in the gathered CB (one face =
// one Tile({1,32}, /*transpose=*/false) page). Sequential accumulate so any
// ring size works, not just even counts.

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);

    uint32_t rt_args_idx = 0;
    const uint32_t has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (has_work == 0) {
        return;
    }

    const uint32_t num_blocks = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    CircularBuffer cb_in(cb_in0);
    CircularBuffer cb_out(cb_out0);

    cb_in.wait_front(num_blocks * block_num_tiles);
    cb_out.reserve_back(block_num_tiles);

    compute_kernel_hw_startup(cb_in0, cb_in0, cb_out0);

    for (uint32_t t = 0; t < block_num_tiles; ++t) {
        copy_init(cb_in0);
        tile_regs_acquire();
        copy_tile(cb_in0, t, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0);
        for (uint32_t block = 1; block < num_blocks; ++block) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0, block * block_num_tiles + t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_out0, t);
        tile_regs_release();
    }

    cb_out.push_back(block_num_tiles);
}
