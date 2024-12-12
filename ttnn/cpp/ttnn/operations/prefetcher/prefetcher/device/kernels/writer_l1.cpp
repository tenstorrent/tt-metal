// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "remote_circular_buffer_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

#include "debug/dprint.h"

FORCE_INLINE uint32_t get_fifo_start_address(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    uint32_t fifo_limit = local_cb.fifo_limit;
    uint32_t fifo_start_addr = fifo_limit - fifo_size;
    return fifo_start_addr;
}

FORCE_INLINE uint32_t get_fifo_start_size(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    return fifo_size;
}

FORCE_INLINE void resize_local_cb_interface(
    uint32_t cb_id, uint32_t page_size, uint32_t fifo_start_addr, uint32_t fifo_start_size) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    uint32_t fifo_limit = local_cb.fifo_limit;

    uint32_t fifo_wr_ptr = local_cb.fifo_wr_ptr;
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_size_page_aligned = fifo_start_size - fifo_start_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_size_page_aligned;
    uint32_t fifo_num_pages = fifo_size_page_aligned / page_size;

    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    // DPRINT << "next_fifo_wr_ptr " <<next_fifo_wr_ptr <<ENDL();
    // DPRINT << "fifo_limit_page_aligned " <<fifo_limit_page_aligned <<ENDL();
    if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        next_fifo_wr_ptr = fifo_start_addr;
    }
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }
    local_cb.fifo_limit = fifo_limit_page_aligned;
    local_cb.fifo_size = fifo_size_page_aligned;
    local_cb.fifo_page_size = page_size;
    local_cb.fifo_num_pages = fifo_num_pages;
    local_cb.fifo_wr_ptr = next_fifo_wr_ptr;
    local_cb.fifo_rd_ptr = next_fifo_rd_ptr;

    // DPRINT << "local_cb.fifo_wr_ptr " << local_cb.fifo_wr_ptr -fifo_start_addr<<ENDL();
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr - fifo_start_addr;
}
FORCE_INLINE uint32_t get_local_cb_wr_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_wr_ptr - fifo_start_addr;
}

FORCE_INLINE void print_remote_fifo(uint32_t cb_id) {
    RemoteSenderCBInterface& cb = get_remote_sender_cb_interface(cb_id);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_sent_ptr + L1_ALIGNMENT);
    // DPRINT << "fifo_wr_ptr " << cb.fifo_wr_ptr << ENDL();
    // DPRINT << "pages_sent " << *pages_sent_ptr << ENDL();
    // DPRINT << "pages_ack " << *pages_acked_ptr << ENDL();

    uint32_t fifo_aligned_num_pages = cb.fifo_limit_page_aligned / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages = fifo_aligned_num_pages - (*pages_sent_ptr - *pages_acked_ptr);
    // DPRINT << "fifo_aligned_num_pages " << fifo_aligned_num_pages << ENDL();
    // DPRINT << "free_pages " << free_pages << ENDL();

    // DPRINT << "fifo_page_size " << cb.fifo_page_size << ENDL();
}

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
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);

    constexpr uint32_t local_cb_id = 0;
    constexpr uint32_t remote_cb_id = 31;

    uint32_t fifo_start_address = get_fifo_start_address(local_cb_id);
    uint32_t fifo_start_size = get_fifo_start_size(local_cb_id);

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
    // const uint32_t noc = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc = noc_index;

    // DPRINT << "num_tensors " << num_tensors <<ENDL();
    // DPRINT << "num_blocks " << num_blocks <<ENDL();

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
            uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_single_tile_sizes = single_tile_sizes[t];
            uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];
            uint32_t curr_block_size = curr_block_num_tiles / num_receivers * curr_single_tile_sizes;

            resize_local_cb_interface(local_cb_id, curr_block_size, fifo_start_address, fifo_start_size);
            experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size, noc);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_wait_front(local_cb_id, 1);

                // DPRINT  << TSLICE(local_cb_id, 0, SliceRange::h0_w0_32(), true, true) << ENDL();

                uint32_t local_cb_addr = get_read_ptr(local_cb_id);
                experimental::remote_cb_reserve_back(remote_cb_id, 1);  // Reserve back 1 curr_block_size

                // print_remote_fifo(remote_cb_id);
                // if (t==0 or t==1 or t==2 or t==3)
                experimental::remote_cb_push_back_and_write_pages(
                    remote_cb_id,
                    local_cb_addr,
                    1,  // wrt to the size of the packet (curr_block_size)
                    curr_block_height_in_tiles,
                    curr_coalesced_num_pages,
                    curr_coalesced_page_size,
                    noc);

                // DPRINT << "block " << block <<ENDL();

                cb_pop_front(local_cb_id, 1);
            }
        }
    }
}
