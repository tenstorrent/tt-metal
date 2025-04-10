// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(5);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(9);
    constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(14);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(15);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(16);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(17);
    constexpr uint32_t output_rows_tiles = get_compile_time_arg_val(18);

    // Split reader args
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(19);
    constexpr uint32_t split_reader = act_block_h_datums != 0;
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(20);
    constexpr uint32_t conv_act_size_c_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t coalesced_read_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t window_outer_offset = get_compile_time_arg_val(23);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(24);
    constexpr uint32_t act_block_h_datums_first_reader = get_compile_time_arg_val(25);
    constexpr uint32_t act_block_h_datums_last_block = get_compile_time_arg_val(26);

    constexpr uint32_t act_block_h_datums_read_last_block =
        act_block_h_datums_last_block > act_block_h_datums
            ? (act_block_h_datums_last_block - act_block_h_datums_first_reader) / 2
            : 0;
    constexpr uint32_t act_block_h_datums_first_reader_read = act_block_h_datums_first_reader / 2;

    uint32_t i = 2;
    const uint32_t out_start_tile_id = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_h = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    i += 1;
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

    constexpr uint32_t act_block_h_datums_read = act_block_h_datums / 2;
    constexpr uint32_t act_block_num_tiles_read = act_block_num_tiles;

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
    uint32_t reader_idx;
    if constexpr (split_reader) {
        packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));
        reader_idx = 0;
    }

// read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(2);
    bool load_bias = true;
#endif

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    const uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        // coalesce reads along weight_size_w
        uint32_t act_l1_offset;
        uint32_t start_reader_idx;
        if constexpr (split_reader) {
            start_reader_idx = 0;
            start_reader_idx = act_block_h_datums_first_reader / 2;
        }

        bool read_weights = true;
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // MCAST RECEIVE WEIGHTS
            // read weight blocks inner dim
            // read weight slice - 1 block of weights in width dim and full weight matrix height
            // read slice only once for all activation blocks

            // TODO: Not sure how this loop works with the additional reader; we don't have a use case for this right
            // now
            for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                 weight_tile_h_outer_i++) {
                uint32_t reader_offset = act_l1_read_addr;
                for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
                    if constexpr (split_reader) {
                        // Do the second half of the reads for act
                        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                        reader_idx = start_reader_idx;
                        cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles_read);
                        uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                        uint32_t act_block_h_datums_read_curr =
                            bh == out_num_blocks_h - 1 ? act_block_h_datums_read_last_block : act_block_h_datums_read;
                        for (uint32_t bhd = 0; bhd < act_block_h_datums_read_curr; bhd++) {
                            // local read from reader_index + reader_offset;
                            uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
                            uint32_t reader_idx_1 = two_reader_indices & 0xffff;
                            uint32_t reader_idx_2 = two_reader_indices >> 16;

                            act_l1_offset = reader_offset + (reader_idx_1 * conv_act_size_c_bytes);
                            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

                            act_l1_offset = reader_offset + (reader_idx_2 * conv_act_size_c_bytes);
                            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

                            reader_idx++;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_id_act_second_reader, act_block_num_tiles_read);

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

                }  // for weight_block_height_num_outer
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
                // Increment reader index for next block in height dim
                start_reader_idx = reader_idx + act_block_h_datums_first_reader_read;
            }
        }  // out_num_blocks_h
    }  // out_num_blocks_w

    cb_wait_front(cb_id_out0, output_rows_tiles);
    noc_async_write_barrier();
}
