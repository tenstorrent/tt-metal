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
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr - fifo_start_addr;
}
FORCE_INLINE uint32_t get_local_cb_wr_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_wr_ptr - fifo_start_addr;
}

FORCE_INLINE uint32_t get_remote_cb_rd_ptr(uint32_t cb_id) {
    RemoteSenderCBInterface& cb = get_remote_sender_cb_interface(cb_id);
    return cb.fifo_wr_ptr;
}

FORCE_INLINE void print_remote_fifo(uint32_t cb_id) {
    RemoteSenderCBInterface& cb = get_remote_sender_cb_interface(cb_id);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_sent_ptr + L1_ALIGNMENT);

    uint32_t fifo_aligned_num_pages = cb.fifo_limit_page_aligned / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages = fifo_aligned_num_pages - (*pages_sent_ptr - *pages_acked_ptr);
}
template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_sender_cb_interface_(uint32_t cb_id, uint32_t page_size, uint8_t noc) {
    ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_cb_interface.config_ptr)[3];
    uint32_t fifo_start_addr = sender_cb_interface.fifo_start_addr;
    uint32_t fifo_wr_ptr = sender_cb_interface.fifo_wr_ptr;
    uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
    uint32_t prev_fifo_limit_page_aligned = sender_cb_interface.fifo_limit_page_aligned;

    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        next_fifo_wr_ptr = fifo_start_addr;
    } else {
#ifndef COMPILE_FOR_TRISC
        if constexpr (update_remote_over_noc) {
            uint32_t aligned_pages_sent_addr = sender_cb_interface.aligned_pages_sent_ptr;
            uint32_t remote_noc_xy_addr = sender_cb_interface.receiver_noc_xy_ptr;
            uint32_t num_receivers = sender_cb_interface.num_receivers;
            uint32_t aligned_page_adjustment =
                (next_fifo_wr_ptr - fifo_wr_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            if (prev_fifo_limit_page_aligned < fifo_limit_page_aligned) {
                aligned_page_adjustment +=
                    (fifo_limit_page_aligned - prev_fifo_limit_page_aligned) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            }
            // increment the aligned pages sent because we skipped to next aligned page location
            volatile tt_l1_ptr uint32_t* pages_sent_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_sent_addr);
            volatile tt_l1_ptr uint32_t* remote_noc_xy_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr);
            for (uint32_t i = 0; i < num_receivers; ++i) {
                uint32_t remote_noc_xy = uint32_t(NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(noc, remote_noc_xy_ptr[0]), DYNAMIC_NOC_Y(noc, remote_noc_xy_ptr[1])));
                *pages_sent_ptr += aligned_page_adjustment;
                uint64_t remote_ack_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)pages_sent_ptr);
                if (aligned_page_adjustment != 0) {
                    noc_semaphore_inc(remote_ack_ptr_addr, aligned_page_adjustment, noc);
                }
                pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
                remote_noc_xy_ptr += 2;

                // DPRINT << "aligned_page_adjustment " << aligned_page_adjustment <<ENDL();
            }
        }
#endif
    }
    sender_cb_interface.fifo_wr_ptr = next_fifo_wr_ptr;
    sender_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    sender_cb_interface.fifo_page_size = page_size;
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
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t local_cb_id = 0;
    constexpr uint32_t remote_cb_id = 31;
    constexpr uint32_t sync_cb = 2;

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

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        DeviceZoneScopedN("layers");
        for (uint32_t t = 0; t < num_tensors; t++) {
            DeviceZoneScopedN("tensors");

            uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
            uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_single_tile_sizes = single_tile_sizes[t];
            uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];
            uint32_t curr_block_size = curr_block_num_tiles * curr_single_tile_sizes;
            uint32_t curr_block_size_per_receiver = curr_block_size / num_receivers;

            resize_remote_sender_cb_interface_<true>(remote_cb_id, curr_block_size_per_receiver, noc);
            {
                DeviceZoneScopedN("reserve_back");
                experimental::remote_cb_reserve_back(remote_cb_id, num_blocks);
            }

            // // To simulate interference
            // if (layer == 1 && t == 1) {
            //     DeviceZoneScopedN("space");
            //     for (volatile int i=0 ; i<16500; ++i);
            // }

            for (uint32_t block = 0; block < num_blocks; ++block) {
                // DeviceZoneScopedN("writer_block");
                cb_wait_front(local_cb_id, max_block_num_tiles);

                uint32_t local_cb_addr = get_read_ptr(local_cb_id);
                experimental::remote_cb_push_back_and_write_pages(
                    remote_cb_id,
                    local_cb_addr,
                    1,  // wrt to the size of the packet (curr_block_size)
                    curr_block_height_in_tiles,
                    curr_coalesced_num_pages,
                    curr_coalesced_page_size,
                    noc);

                cb_pop_front(local_cb_id, max_block_num_tiles);
            }
        }
    }
}
