// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "remote_circular_buffer_api.h"
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

// FORCE_INLINE uint32_t get_remote_fifo_start_address(uint32_t cb_id) {
//     RemoteReceiverCBInterface& cb = get_remote_receiver_cb_interface(cb_id);
//     return cb.fifo_start_addr;
// }

// FORCE_INLINE uint32_t get_remote_fifo_start_size(uint32_t cb_id) {
//     RemoteReceiverCBInterface& cb = get_remote_receiver_cb_interface(cb_id);
//     uint32_t fifo_limit_page_aligned = cb.fifo_limit_page_aligned;
//     return fifo_limit_page_aligned - cb.fifo_start_addr;
// }
FORCE_INLINE uint32_t get_remote_cb_rd_ptr(uint32_t cb_id) {
    RemoteReceiverCBInterface& cb = get_remote_receiver_cb_interface(cb_id);
    return cb.fifo_rd_ptr;
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr;
}
FORCE_INLINE uint32_t get_local_cb_wr_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_wr_ptr;
}

template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_receiver_cb_interface_(uint32_t cb_id, uint32_t page_size, uint8_t noc) {
    ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.config_ptr)[3];
    uint32_t fifo_start_addr = receiver_cb_interface.fifo_start_addr;
    uint32_t fifo_rd_ptr = receiver_cb_interface.fifo_rd_ptr;
    uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

    // DPRINT << "fifo_rd_ptr " << fifo_rd_ptr << ENDL();
    // DPRINT << "page_size " << page_size << ENDL();
    // DPRINT << "fifo_limit_page_aligned " << fifo_limit_page_aligned << ENDL();
    // DPRINT << "fifo_start_addr " << fifo_start_addr << ENDL();
    // DPRINT << "fifo_rd_ptr - fifo_start_addr " << fifo_rd_ptr - fifo_start_addr << ENDL();
    // DPRINT << "align(fifo_rd_ptr - fifo_start_addr, page_size) " <<align(fifo_rd_ptr - fifo_start_addr, page_size)<<
    // ENDL();

    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    // DPRINT << "next_fifo_rd_ptr " << next_fifo_rd_ptr << ENDL();

    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    } else {
#ifndef COMPILE_FOR_TRISC
        if constexpr (update_remote_over_noc) {
            uint32_t aligned_pages_acked_addr = receiver_cb_interface.aligned_pages_acked_ptr;
            uint32_t sender_noc_x = receiver_cb_interface.sender_noc_x;
            uint32_t sender_noc_y = receiver_cb_interface.sender_noc_y;
            uint32_t aligned_page_adjustment =
                (next_fifo_rd_ptr - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            // increment the aligned pages acked because we skipped to next aligned page location
            volatile tt_l1_ptr uint32_t* pages_acked_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_acked_addr);
            *pages_acked_ptr += aligned_page_adjustment;
            uint64_t remote_ack_ptr_addr = get_noc_addr(sender_noc_x, sender_noc_y, (uint32_t)pages_acked_ptr, noc);
            noc_semaphore_inc(remote_ack_ptr_addr, aligned_page_adjustment, noc);
        }
#endif
    }
    receiver_cb_interface.fifo_rd_ptr = next_fifo_rd_ptr;
    receiver_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    receiver_cb_interface.fifo_page_size = page_size;
}

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);

    // DPRINT << "AAAAAAAA " << ENDL();

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t remote_cb_id = 31;
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

    uint32_t fifo_start_address = get_fifo_start_address(cb_id_in1);
    uint32_t fifo_start_size = get_fifo_start_size(cb_id_in1);

    // DPRINT << "fifo_start_address " << fifo_start_address << ENDL();
    // DPRINT << "fifo_start_size " << fifo_start_size << ENDL();

    // uint32_t fifo_remote_start_address = get_remote_fifo_start_address(remote_cb_id);
    // uint32_t fifo_remote_start_size = get_remote_fifo_start_size(remote_cb_id);

    // DPRINT << "fifo_remote_start_address " << fifo_remote_start_address << ENDL();
    // DPRINT << "fifo_remote_start_size " << fifo_remote_start_size << ENDL();

    // DPRINT << "curr_block_size " << in1_block_num_tiles * get_tile_size(cb_id_in1) << ENDL();

    // experimental::setup_remote_cb_interfaces<true>(cb_l1_base, end_cb_index);
    // resize_remote_receiver_cb_interface_<true>(remote_cb_id, in1_block_num_tiles * get_tile_size(cb_id_in1),
    // noc_index); experimental::align_local_cbs_to_remote_cb<1>(remote_cb_id, {cb_id_in1});

    constexpr uint32_t sync_cb = 5;
    // cb_reserve_back(sync_cb, 1);
    // cb_push_back(sync_cb, 1);

#ifdef ENABLE_GLOBAL_CB
    uint32_t in1_num_blocks_wait = in1_block_num_tiles * ring_idx;
#endif
    // DPRINT << "in1 num_blocks " << num_blocks << ENDL();
    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(cb_id_in1, shard_size_in_tiles);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif
        // DPRINT << get_remote_cb_rd_ptr(remote_cb_id) << ENDL();
        LocalCBInterface& local_cb = get_local_cb_interface(cb_id_in1);
        // DPRINT << "reader " << get_tile_size(cb_id_in1) << ENDL();
        // DPRINT << "reader " << in1_block_num_tiles << ENDL();
        // DPRINT << "reader " << local_cb.fifo_size << ENDL();
        // DPRINT << "reader " << local_cb.fifo_limit << ENDL();
        // DPRINT << "reader " << get_local_cb_rd_ptr(cb_id_in1) << ENDL();
        // DPRINT << TSLICE(cb_id_in1, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, true)
        // << ENDL(); DPRINT << TSLICE(cb_id_in1, 0, SliceRange{.h0 = 1, .h1 = 2, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
        // true, true) << ENDL(); DPRINT << TSLICE(cb_id_in1, 0, SliceRange{.h0 = 2, .h1 = 3, .hs = 1, .w0 = 0, .w1 =
        // 32, .ws = 1}, true, true) << ENDL();

        // for (int i=0; i<36;i++)
        // DPRINT  << TSLICE(cb_id_in1, i, SliceRange::h0_w0_32(), true, true) << ENDL();
        // DPRINT << "reader " << get_local_cb_wr_ptr(cb_id_in1) << ENDL();
        // cb_push_back(cb_id_in1, 1);
        // DPRINT << "reader " << get_local_cb_wr_ptr(cb_id_in1) << ENDL();

        // DPRINT << "get_local_cb_rd_ptr " << get_local_cb_rd_ptr(cb_id_in1) << ENDL();

        cb_push_back(cb_id_in1, shard_size_in_tiles);

#ifdef ENABLE_GLOBAL_CB
        // cb_reserve_back(cb_id_in1, in1_num_blocks_wait);
        // in1_num_blocks_wait += in1_block_num_tiles;
        for (uint32_t block = 0; block < num_blocks; block++) {
            // cb_reserve_back(cb_id_in1, in1_num_blocks_wait);
            // in1_num_blocks_wait += in1_block_num_tiles;
            cb_wait_front(sync_cb, 1);
            experimental::remote_cb_pop_front(remote_cb_id, 1);
            cb_pop_front(sync_cb, 1);
        }

#endif

        // DPRINT << "k " << get_remote_cb_rd_ptr(remote_cb_id) << ENDL();

        // experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    // DPRINT << "get_local_cb_rd_ptr " << get_local_cb_rd_ptr(cb_id_in1) << ENDL();
    // experimental::align_local_cbs_to_remote_cb<1>(remote_cb_id, {cb_id_in1});
    // DPRINT << "get_local_cb_rd_ptr " << get_local_cb_rd_ptr(cb_id_in1) << ENDL();

    DPRINT << "in1 DONE" << ENDL();
}
