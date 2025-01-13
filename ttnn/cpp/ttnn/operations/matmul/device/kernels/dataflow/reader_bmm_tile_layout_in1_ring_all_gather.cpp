// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "remote_circular_buffer_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t sync_cb = tt::CBIndex::c_3;
    constexpr uint32_t sync_cb2 = tt::CBIndex::c_4;
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(sync_cb2, 1);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif

        cb_push_back(sync_cb2, 1);

        RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(remote_cb_id);

        // DPRINT  << "cb_id_in1_addr before : " << (uint)(remote_cb.fifo_rd_ptr / 16) << ENDL();

#ifdef ENABLE_GLOBAL_CB
        cb_wait_front(sync_cb, 1);
        experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
        cb_pop_front(sync_cb, 1);

        // cb_wait_front(tt::CBIndex::c_5, in1_block_num_tiles);
        // for (uint32_t i=0; i<16; ++i) {
        //     DPRINT  << "i " << i << ENDL();
        //     for (uint32_t j=0; j<32; ++j)
        //         DPRINT  << TSLICE(tt::CBIndex::c_5, i, SliceRange{.h0 = uint8_t(j), .h1 = uint8_t(j+1), .hs = 1, .w0
        //         = 0, .w1 = 32, .ws = 1}, true, true) << ENDL();
        // }
        uint32_t aligned_pages_acked_addr = remote_cb.aligned_pages_acked_ptr;
        volatile tt_l1_ptr uint32_t* pages_acked_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_acked_addr);
        DPRINT << "pages acked : " << (uint)(*pages_acked_ptr) << ENDL();
#endif
    }

#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
#endif
}
