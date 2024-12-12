// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "remote_circular_buffer_api.h"
#include "debug/dprint.h"

// FORCE_INLINE uint32_t get_fifo_start_address(uint32_t cb_id) {
//     LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
//     uint32_t fifo_size = local_cb.fifo_size;
//     uint32_t fifo_limit = local_cb.fifo_limit;
//     uint32_t fifo_start_addr = fifo_limit - fifo_size;
//     return fifo_start_addr;
// }

// FORCE_INLINE uint32_t get_fifo_start_size(uint32_t cb_id) {
//     LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
//     uint32_t fifo_size = local_cb.fifo_size;
//     return fifo_size;
// }

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

// FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
//     LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
//     return local_cb.fifo_rd_ptr - fifo_start_addr;
// }
// FORCE_INLINE uint32_t get_local_cb_wr_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
//     LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
//     return local_cb.fifo_wr_ptr - fifo_start_addr;
// }

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

    // uint32_t fifo_start_address = get_fifo_start_address(cb_id_in1);
    // uint32_t fifo_start_size = get_fifo_start_size(cb_id_in1);

    // DPRINT << "fifo_start_address " << fifo_start_address << ENDL();
    // DPRINT << "fifo_start_size " << fifo_start_size << ENDL();

    // uint32_t fifo_remote_start_address = get_remote_fifo_start_address(remote_cb_id);
    // uint32_t fifo_remote_start_size = get_remote_fifo_start_size(remote_cb_id);

    // DPRINT << "fifo_remote_start_address " << fifo_remote_start_address << ENDL();
    // DPRINT << "fifo_remote_start_size " << fifo_remote_start_size << ENDL();

    // DPRINT << "curr_block_size " << in1_block_num_tiles * get_tile_size(cb_id_in1) << ENDL();

    // experimental::resize_remote_receiver_cb_interface(remote_cb_id, in1_block_num_tiles * get_tile_size(cb_id_in1),
    // noc_index); experimental::align_local_cbs_to_remote_cb<1>(remote_cb_id, {cb_id_in1});

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
        // DPRINT << get_local_cb_rd_ptr(cb_id_in1, fifo_start_address) << ENDL();
        // DPRINT << TSLICE(cb_id_in1, 0, SliceRange::h0_w0_32(), true, true) << ENDL();

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

        // DPRINT << "k " << get_remote_cb_rd_ptr(remote_cb_id) << ENDL();
    }
}
