// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t read_cb_size = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t max_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t addrs_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(8);
    constexpr bool skip_ptr_update = get_compile_time_arg_val(9);

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t total_num_blocks_in_buffer = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    uint32_t l1_buffer_start_addr = get_write_ptr(cb_id);
    uint32_t l1_buffer_end_addr = get_write_ptr(cb_id) + read_cb_size;

    volatile tt_l1_ptr uint32_t* tensor_addrs_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(addrs_cb_id));

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_page_size = page_sizes[t];
            uint32_t curr_block_num_pages = block_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];

            // Address setup
            uint32_t tensor_base_address = tensor_addrs_l1[layer * num_tensors + t];
            uint32_t src_base_addr =
                noc_async_read_tile_dram_sharded_set_state<true>(tensor_base_address, curr_page_size, bank_id, vc);
            uint32_t src_read_addr = 0;

            uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            cb_reserve_back(cb_id, max_block_num_tiles);

            uint32_t l1_write_addr_offset = 0;
            uint32_t l1_write_addr_start = get_write_ptr(cb_id);
            // Wrap around l1_write_addr if it reaches l1_buffer_end_addr
            if (l1_write_addr_start >= l1_buffer_end_addr) {
                l1_write_addr_start = l1_buffer_start_addr;
            }
            uint32_t l1_write_addr = l1_write_addr_start;

            for (uint32_t block = 0; block < num_blocks; block++) {
                // Set trid for current block
                noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

                // Issue noc async read commands for current block
                uint32_t temp_l1_write_addr = l1_write_addr;
                for (uint32_t h = 0; h < curr_block_num_pages; ++h) {
                    noc_async_read_tile_dram_sharded_with_state_with_trid<skip_ptr_update>(
                        src_base_addr, src_read_addr, temp_l1_write_addr, curr_block_trid);
                    src_read_addr += curr_page_size;
                    temp_l1_write_addr += curr_page_size;
                }

                if (num_free_blocks_in_buffer == 3) {  // After first block we don't block but continue issuing reads
                    num_free_blocks_in_buffer -= 1;
                } else {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(cb_id, max_block_num_tiles);
                    block_trid_to_wait =
                        block_trid_to_wait == total_num_blocks_in_buffer ? 1 : (block_trid_to_wait + 1);
                }

                // We still have blocks to read
                if (block != num_blocks - 1) {
                    // Increment block_trid, wrap around to 1 if it reaches total_num_blocks_in_buffer
                    curr_block_trid = curr_block_trid == total_num_blocks_in_buffer ? 1 : (curr_block_trid + 1);

                    // Wrap around l1_write_addr if it reaches l1_buffer_end_addr
                    l1_write_addr += max_block_size;
                    if (l1_write_addr >= l1_buffer_end_addr) {
                        l1_write_addr = l1_buffer_start_addr;
                    }

                    cb_reserve_back(
                        cb_id,
                        max_block_num_tiles *
                            2);  // Reserve two blocks of spcae to issue multiple block reads in parallel
                }
            }

            // last block to wait
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id, max_block_num_tiles);
        }
    }

    // wait for signal to exit, since reader cannot exit early due to the ongoing traffic on the same noc.
    cb_wait_front(sync_cb_id, 1);
    cb_pop_front(sync_cb_id, 1);

    // reset noc counters here because we didn't properly update ptrs for better perf.
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }
}
