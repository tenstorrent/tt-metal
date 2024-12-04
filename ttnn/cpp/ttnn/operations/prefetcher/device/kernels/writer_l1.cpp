// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "remote_circular_buffer_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

#include "debug/dprint.h"

constexpr uint32_t noc = get_compile_time_arg_val(0);
constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
constexpr uint32_t num_receivers = get_compile_time_arg_val(2);
constexpr uint32_t num_tensors = get_compile_time_arg_val(3);
constexpr uint32_t local_cb_id = get_compile_time_arg_val(4);
constexpr uint32_t remote_cb_id = get_compile_time_arg_val(5);

tt_l1_ptr uint32_t* coalesced_page_size;
tt_l1_ptr uint32_t* coalesced_num_pages;
tt_l1_ptr uint32_t* block_num_tiles;
tt_l1_ptr uint32_t* single_tile_sizes;
tt_l1_ptr uint32_t* num_tile_rows;

uint32_t layer = 0;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    coalesced_page_size = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    coalesced_num_pages = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    block_num_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    single_tile_sizes = (tt_l1_ptr uint32_t*)(get_arg_addr(
        increment_arg_idx(rt_args_idx, num_tensors)));  // why is this page_size and not single_tile_size??
    num_tile_rows = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    for (uint32_t l = 0; l < num_tensors; ++l) {
        uint32_t curr_coalesced_page_size = coalesced_page_size[l];
        uint32_t curr_coalesced_num_pages = coalesced_num_pages[l];
        uint32_t curr_block_num_tiles = block_num_tiles[l];
        uint32_t curr_single_tile_size = single_tile_sizes[l];
        uint32_t curr_num_tile_rows = num_tile_rows[l];
        uint32_t curr_receiver_block_num_tiles = curr_block_num_tiles / num_receivers;

        uint32_t curr_block_size = curr_receiver_block_num_tiles * curr_single_tile_size;
        experimental::resize_remote_sender_cb_interface(remote_cb_id, curr_block_size, noc_index);

        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(local_cb_id, curr_block_num_tiles);

            uint32_t local_cb_addr = get_read_ptr(local_cb_id);
            experimental::remote_cb_reserve_back(remote_cb_id, 1);
            experimental::remote_cb_push_back_and_write_pages(
                remote_cb_id,
                local_cb_addr,
                1,
                curr_num_tile_rows,
                curr_coalesced_num_pages,
                curr_coalesced_page_size,
                noc);

            cb_pop_front(local_cb_id, curr_block_num_tiles);
        }
        layer++;
    }
}
