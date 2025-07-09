// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);
    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);
    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i++);
    if (noop) {
        return;
    }

    // mcast args
    const uint32_t weights_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));

    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    const uint64_t weights_mcast_sender_semaphore_noc_addr =
        get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);

// read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    bool load_bias = true;
#endif

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // MCAST RECEIVE WEIGHTS
            // read weight blocks inner dim
            // read weight slice - 1 block of weights in width dim and full weight matrix height
            // read slice only once for all activation blocks
            for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                 weight_tile_h_outer_i++) {
                cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                // Set weights semaphore value to INVALID
                noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

                cb_push_back(cb_id_weight, weight_block_num_tiles);
            }  // for weight_block_height_num_outer

#ifdef FUSE_BIAS
            if (load_bias) {
                cb_reserve_back(bias_cb_id, bias_ntiles);

                // Set weights semaphore value to INVALID
                noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

                cb_push_back(bias_cb_id, bias_ntiles);
                load_bias = false;
            }
#endif

        }  // out_num_blocks_h
    }  // out_num_blocks_w

    noc_async_write_barrier();
}
