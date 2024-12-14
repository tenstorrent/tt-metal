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

FORCE_INLINE void print_remote_fifo(uint32_t cb_id) {
    RemoteReceiverCBInterface& cb = get_remote_receiver_cb_interface(cb_id);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.aligned_pages_acked_ptr - L1_ALIGNMENT);

    uint32_t fifo_aligned_num_pages = cb.fifo_limit_page_aligned / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages = fifo_aligned_num_pages - (*pages_sent_ptr - *pages_acked_ptr);
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

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t remote_cb_id = 31;
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

    uint32_t fifo_start_address = get_fifo_start_address(cb_id_in1);
    uint32_t fifo_start_size = get_fifo_start_size(cb_id_in1);

    constexpr uint32_t sync_cb = 5;
    constexpr uint32_t sync_cb2 = 6;

#ifdef ENABLE_GLOBAL_CB
    uint32_t in1_num_blocks_wait = in1_block_num_tiles * ring_idx;
#endif
    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(sync_cb2, 1);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif
        LocalCBInterface& local_cb = get_local_cb_interface(cb_id_in1);

        cb_push_back(sync_cb2, 1);

#ifdef ENABLE_GLOBAL_CB
        cb_wait_front(sync_cb, 1);
        experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
        cb_pop_front(sync_cb, 1);
#endif
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
}
