// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// Publishes this core's operands and, level by level, the partials its children write into
// cb_recv. The only synchronization is with this core's own children: semaphore l counts the
// level-l children whose partial has landed.
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_index = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t recv_cb_index = get_named_compile_time_arg_val("cb_recv");

    constexpr uint32_t in0_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_levels = get_compile_time_arg_val(3);

    // Both CBs alias the L1-sharded operands, so the data is already in place.
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    in0_cb.reserve_back(in0_num_tiles);
    in0_cb.push_back(in0_num_tiles);
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    CircularBuffer recv_cb(recv_cb_index);
    for (uint32_t level = 0; level < num_levels; ++level) {
        // A core collects children on levels 0 .. L-1 for some L and on none above.
        const uint32_t num_children = get_arg_val<uint32_t>(level);
        if (num_children == 0) {
            break;
        }
        const uint32_t level_num_tiles = num_children * block_num_tiles;
        recv_cb.reserve_back(level_num_tiles);
        Semaphore<> arrived(level);
        arrived.wait(num_children);
        arrived.set(0);
        recv_cb.push_back(level_num_tiles);
    }
}
