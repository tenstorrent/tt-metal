// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "remote_circular_buffer_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t remote_cb_id = 31;
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

#ifdef ENABLE_GLOBAL_CB
    uint32_t in1_num_blocks_wait = in1_block_num_tiles * ring_idx;
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(cb_id_in1, shard_size_in_tiles);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif
        cb_push_back(cb_id_in1, shard_size_in_tiles);

#ifdef ENABLE_GLOBAL_CB
        cb_reserve_back(cb_id_in1, in1_num_blocks_wait);
        in1_num_blocks_wait += in1_block_num_tiles;
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_reserve_back(cb_id_in1, in1_num_blocks_wait);
            in1_num_blocks_wait += in1_block_num_tiles;
            experimental::remote_cb_pop_front(remote_cb_id, 1);
        }
#endif
    }
}
