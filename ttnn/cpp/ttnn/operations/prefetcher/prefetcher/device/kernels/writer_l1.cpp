// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "remote_circular_buffer_api.h"
#include "debug/dprint.h"

uint32_t increment_arg_idx(uint32_t& arg_idx, uint32_t num_args = 1) {
    uint32_t old_arg_idx = arg_idx;
    arg_idx += num_args;
    return old_arg_idx;
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t local_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(7);
    constexpr bool skip_ptr_update = get_compile_time_arg_val(8);

    // Runtime args
    // Note: Coalesced sizes -> wrt to receiver cores, sizes -> wrt to dram reader cores
    uint32_t rt_args_idx = 0;
    const uint32_t* coalesced_page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* coalesced_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* single_tile_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_height_in_tiles =
        (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));  // Kt / num_blocks = in_block_h;

    uint32_t noc = noc_index;
    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
            uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_single_tile_sizes = single_tile_sizes[t];
            uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];
            uint32_t curr_block_size = curr_block_num_tiles * curr_single_tile_sizes;
            uint32_t curr_block_size_per_receiver = curr_block_size / num_receivers;

            experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                {
                    cb_wait_front(local_cb_id, max_block_num_tiles);
                    experimental::remote_cb_reserve_back(remote_cb_id, 1);

                    uint32_t local_cb_addr = get_read_ptr(local_cb_id);
                    experimental::remote_cb_push_back_and_write_pages<skip_ptr_update>(
                        remote_cb_id,
                        local_cb_addr,
                        1,  // wrt to the size of the packet (curr_block_size)
                        curr_block_height_in_tiles,
                        curr_coalesced_num_pages,
                        curr_coalesced_page_size,
                        noc);

                    noc_async_posted_writes_flushed();

                    cb_pop_front(local_cb_id, max_block_num_tiles);
                }
            }
        }
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);

    // reset noc counters here because we didn't properly update ptrs for better perf.
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }

    // signal reader can exit, since reader cannot exit early due to the ongoing traffic on the same noc.
    cb_reserve_back(sync_cb_id, 1);
    cb_push_back(sync_cb_id, 1);
}
