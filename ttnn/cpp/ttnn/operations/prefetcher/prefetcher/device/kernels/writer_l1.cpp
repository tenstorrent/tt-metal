// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "remote_circular_buffer_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

#include "debug/dprint.h"

uint32_t layer = 0;

/*
TODO:
    - How do the coalesced sizes differ?
    - Mimick use of get_max_page_size_and_num_pages for the reader kernel,
    but change the num_tiles to only be the size of one row
    - Add num_layers as CT arg (and the for-loop)
    - block_num_pages is the same as in reader (so, block_num_pages)

*/
void kernel_main() {
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(2);

    constexpr uint32_t local_cb_id = 0;   // TODO: Set this
    constexpr uint32_t remote_cb_id = 0;  // TODO: Set this

    /*
        Note: Coalesced sizes -> wrt to receiver cores
              sizes -> wrt to dram reader cores
    */

    uint32_t rt_args_idx = 0;
    const uint32_t* coalesced_page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* coalesced_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* single_tile_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_height_in_tiles =
        (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));  // Kt / num_blocks = in_block_h;
    const uint32_t noc = get_arg_val(rt_args_idx++);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
            uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_single_tile_sizes = single_tile_sizes[t];
            uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];
            uint32_t curr_block_size = curr_block_num_tiles / num_receivers * curr_single_tile_sizes;

            experimental::resize_remote_sender_cb_interface(remote_cb_id, curr_block_size, noc);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_wait_front(local_cb_id, curr_block_num_tiles);

                uint32_t local_cb_addr = get_read_ptr(local_cb_id);
                experimental::remote_cb_reserve_back(remote_cb_id, 1);
                experimental::remote_cb_push_back_and_write_pages(
                    remote_cb_id,
                    local_cb_addr,
                    1,  // wrt to the size of the packet
                    curr_block_height_in_tiles,
                    curr_coalesced_num_pages,
                    curr_coalesced_page_size,
                    noc);

                cb_pop_front(local_cb_id, curr_block_num_tiles);
            }
        }
    }
}
