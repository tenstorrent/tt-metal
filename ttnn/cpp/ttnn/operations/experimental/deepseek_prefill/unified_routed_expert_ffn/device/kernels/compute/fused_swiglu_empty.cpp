// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DIAGNOSTIC: compute drains the phase-1 reader output (14 K-blocks of
// in0_x + in1_gate) and pushes a single tile to cb_out.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);

    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    for (uint32_t kb = 0; kb < g_num_blocks; ++kb) {
        cb_wait_front(cb_in0_x, g_in0_block_num_tiles);
        cb_wait_front(cb_in1_gate, g_in1_block_num_tiles);
        cb_pop_front(cb_in0_x, g_in0_block_num_tiles);
        cb_pop_front(cb_in1_gate, g_in1_block_num_tiles);
    }

    cb_reserve_back(cb_out, 1);
    cb_push_back(cb_out, 1);
}
