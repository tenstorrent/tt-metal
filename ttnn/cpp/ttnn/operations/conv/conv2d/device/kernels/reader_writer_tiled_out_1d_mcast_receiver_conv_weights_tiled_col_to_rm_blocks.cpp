// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "conv_reader_common.hpp"
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

    constexpr bool fuse_bias = get_compile_time_arg_val(18);

    constexpr bool split_reader_enabled = get_compile_time_arg_val(19);
    constexpr bool activation_reuse_enabled = get_compile_time_arg_val(20);

    // Split reader args
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(23);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(24);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(25);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(27);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(28);
    constexpr uint32_t stride_w = get_compile_time_arg_val(29);
    constexpr uint32_t weights_size_h = get_compile_time_arg_val(30);

    // Activation reuse args
    constexpr uint32_t act_reuse_cb_tiles = get_compile_time_arg_val(31);
    constexpr uint32_t act_block_w_tiles = get_compile_time_arg_val(32);
    constexpr bool readers_process_full_image_widths = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t image_width_tiles = get_compile_time_arg_val(34);
    constexpr uint32_t output_image_width = get_compile_time_arg_val(35);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(36);
    constexpr bool need_to_push_remaining_tiles = get_compile_time_arg_val(37) == 1;
    constexpr bool single_core_processes_multiple_batches = get_compile_time_arg_val(38) == 1;

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i++);

    if (noop) {
        return;
    }

    if constexpr (split_reader_enabled) {
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<cb_id_act_second_reader>();
        }
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

    const uint32_t remaining_tiles_to_push =
        split_reader_enabled && activation_reuse_enabled ? get_arg_val<uint32_t>(i++) : 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_wait_front(cb_reader_indices, 1);
#endif
    }
    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        split_reader_enabled ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices))
                             : nullptr;
    uint32_t reader_idx = 0;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    const uint32_t act_l1_read_addr = split_reader_enabled ? get_read_ptr(cb_id_sharded_act) : 0;
    uint32_t start_reader_idx =
        split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1 : 0;
    const uint32_t cb_start_addr = split_reader_enabled ? get_write_ptr(cb_id_act_second_reader) : 0;

    // read in bias if enabled (done only once for all batches)
    bool load_bias = true;
    uint32_t l1_write_addr_act = 0;
    uint32_t reader_offset = 0;
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // MCAST RECEIVE WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks

        if constexpr (split_reader_enabled) {
            if constexpr (activation_reuse_enabled) {
                l1_write_addr_act = cb_start_addr;
                get_local_cb_interface(cb_id_act_second_reader).fifo_wr_ptr = l1_write_addr_act;
            }
            reader_offset = act_l1_read_addr;
        }
        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
            if constexpr (split_reader_enabled) {
                // Do the second half of the reads for act
                noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                reader_idx = start_reader_idx;

                if constexpr (!activation_reuse_enabled) {
                    cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles);
                    l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
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
                } else {
                    read_sticks_activation_reuse<
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        window_outer_offset,
                        weight_size_w,
                        stride_w,
                        weights_size_h,
                        cb_id_act_second_reader,
                        act_reuse_cb_tiles,
                        act_block_w_tiles,
                        readers_process_full_image_widths,
                        image_width_tiles,
                        output_image_width,
                        window_reuse_offset,
                        single_core_processes_multiple_batches>(
                        packed_reader_indices_ptr, act_l1_read_addr, l1_write_addr_act, reader_idx, cb_start_addr);

                    if constexpr (need_to_push_remaining_tiles) {
                        if (block_weight_h == num_blocks_weight_h - 1) {
                            // Last core sometimes has less work to do, but we still need to push the same number of
                            // tiles to avoid blocking compute kernels
                            push_remaining_tiles<cb_id_act_second_reader, act_block_w_tiles, image_width_tiles>(
                                remaining_tiles_to_push, cb_start_addr);
                        }
                    }
                }
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

        if constexpr (fuse_bias) {
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
        }

        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
    }  // out_num_blocks_h

    noc_async_write_barrier();
}
