// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "height_sharded_reader_common.hpp"
#include "debug/debug.h"

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);

    // Split reader args
    constexpr bool split_reader = get_compile_time_arg_val(17);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(20);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(21);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(22);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(24);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(25);
    constexpr uint32_t stride_w = get_compile_time_arg_val(26);

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i++);

    if (noop) {
        return;
    }

    if constexpr (split_reader && needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_second_reader>();
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;

    // mcast args
    const uint32_t weights_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));

    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    const uint64_t weights_mcast_sender_semaphore_noc_addr =
        get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
    uint32_t reader_idx;
    if constexpr (split_reader) {
        packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));
        reader_idx = 0;
    }

// read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    bool load_bias = true;
#endif

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    const uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    // coalesce reads along weight_size_w
    uint32_t start_reader_idx;
    if constexpr (split_reader) {
        start_reader_idx = (uint32_t)(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
    }
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // MCAST RECEIVE WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks

        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
            if constexpr (split_reader) {
                // Do the second half of the reads for act
                noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                reader_idx = start_reader_idx;
                cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles);
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                read_sticks<
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w,
                    stride_w>(packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
                noc_async_read_barrier();
                cb_push_back(cb_id_act_second_reader, act_block_num_tiles);

                reader_offset += window_outer_offset;
            }

            // Receive weights
            cb_reserve_back(cb_id_weight, weight_block_num_tiles);
            if (bh == 0) {
                // Set weights semaphore value to INVALID
                noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts
                // data)
                noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);
            }

            cb_push_back(cb_id_weight, weight_block_num_tiles);
        }

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

        if constexpr (split_reader) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
    }  // out_num_blocks_h

    noc_async_write_barrier();
}
