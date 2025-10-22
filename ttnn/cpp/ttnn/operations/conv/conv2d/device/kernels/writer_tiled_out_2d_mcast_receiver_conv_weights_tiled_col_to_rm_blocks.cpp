// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "conv_reader_common.hpp"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);
    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);

#ifdef SPLIT_READER
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t window_outer = get_compile_time_arg_val(6);  // num_blocks_act_w
    constexpr bool sliced_inner_dim = window_outer > 1;             // Derived like block sharded reader
    constexpr uint32_t act_block_num_tiles_split_last = get_compile_time_arg_val(18);  // This is what factory passes
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(20);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(21);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(22);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(24);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(25);
    constexpr uint32_t stride_w = get_compile_time_arg_val(26);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(27);  // Input filter window height

    // Split multicasting compile-time args (always present when split reader is enabled)
    const uint32_t act_first_reader_tilize_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(28));
    const uint32_t act_second_reader_mcast_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(29));
    const uint32_t act_mcast_sender_semaphore_addr_second = get_semaphore(get_compile_time_arg_val(30));
    const uint32_t act_mcast_receiver_semaphore_addr_second = get_semaphore(get_compile_time_arg_val(31));
    constexpr uint32_t act_mcast_sender_size_bytes_second = get_compile_time_arg_val(32);
    constexpr uint32_t act_mcast_first_offset = get_compile_time_arg_val(33);
    constexpr uint32_t act_mcast_second_offset = get_compile_time_arg_val(34);

    constexpr bool transpose_mcast = get_compile_time_arg_val(35);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(36);
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(37);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(38);
    constexpr uint32_t act_mcast_num_dests = get_compile_time_arg_val(39);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(40);

    volatile tt_l1_ptr uint32_t* act_first_reader_tilize_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_first_reader_tilize_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_second_reader_mcast_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_second_reader_mcast_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr_second =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr_second);
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr_second =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr_second);

#ifdef SPLIT_READER_CB_SHARED
    // When the split reader CB is shared, both readers write to the same circular buffer.
    // Synchronization is required: the main reader signals when CB space is reserved,
    // and the second reader signals when it has finished writing its portion.
    constexpr bool split_reader_cb_shared = true;
    const uint32_t act_split_reader_reserve_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(41));
    const uint32_t act_split_reader_write_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(42));
    constexpr uint32_t act_write_offset = get_compile_time_arg_val(43);
    constexpr uint32_t act_write_offset_last = get_compile_time_arg_val(44);

    volatile tt_l1_ptr uint32_t* act_split_reader_reserve_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_reserve_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_split_reader_write_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_write_done_semaphore_addr);
#else
    constexpr bool split_reader_cb_shared = false;
    constexpr uint32_t act_write_offset = 0;
    constexpr uint32_t act_write_offset_last = 0;

    volatile tt_l1_ptr uint32_t* act_split_reader_reserve_done_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* act_split_reader_write_done_semaphore_addr_ptr = nullptr;
#endif

    const uint32_t split_reader_cb_write_addr = get_write_ptr(cb_id_act_second_reader) + act_write_offset;
    // In case of double buffering the split reader can write to two different addresses
    const uint32_t split_reader_cb_write_addr_last = get_write_ptr(cb_id_act_second_reader) + act_write_offset_last;
    const uint32_t split_reader_cb_write_addr_sum = split_reader_cb_write_addr + split_reader_cb_write_addr_last;
#else
    const uint32_t split_reader_cb_write_addr = 0;
#endif

    // mcast args
    uint32_t i = 0;
    const uint32_t weights_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_receiver_core_act = get_arg_val<uint32_t>(i++) > 0;

    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    const uint64_t weights_mcast_sender_semaphore_noc_addr =
        get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);

#ifdef SPLIT_READER
    uint32_t act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_id = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    tt_l1_ptr uint32_t* act_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

#ifdef CONFIG_TENSOR_IN_DRAM
    cb_wait_front(cb_reader_indices, 1);
#endif
    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_second_reader>();
    }
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // Initial setup for second reader (starting from second reader's data)
    uint32_t start_reader_idx = (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1;
    uint32_t reader_idx = start_reader_idx;

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;

    const uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    // Activation multicasting for split reader's second half
    // L1 array for semaphore valid value
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    l1_array[0] = 1;  // Load const 1 to be used as semaphore valid value
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);

    // Set up NOC addresses
    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x, act_mcast_dest_noc_start_y, act_mcast_dest_noc_end_x, act_mcast_dest_noc_end_y, 0);
    uint64_t act_mcast_receiver_semaphore_noc_addr_second =
        act_multicast_noc_addr | act_mcast_receiver_semaphore_addr_second;

    // Reset receiver semaphore to VALID
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr_second, VALID);

    // Calculate the number of activation blocks to multicast
    constexpr uint32_t act_w_num_outer = window_outer;  // Same as conv_act_c_blocks in reader

    // Calculate addresses for second half
    uint32_t tilized_act_start_address = get_read_ptr(tilized_in0_cb_id) + act_mcast_first_offset;
    uint32_t act_mcast_offset = act_mcast_first_offset;
    constexpr uint32_t act_mcast_offset_sum = act_mcast_first_offset + act_mcast_second_offset;
#endif
// read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    bool load_bias = true;
#endif

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t l1_write_addr_act = split_reader_cb_write_addr;
    uint32_t prev_addr = 0;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
#ifdef SPLIT_READER
            // Read activation data using block sharded pattern (for second reader)
            uint32_t reader_offset = act_l1_read_addr;
#endif
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
#ifdef SPLIT_READER
                reader_idx = start_reader_idx;
                if constexpr (!split_reader_cb_shared) {
                    cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                }

                if (is_sender_core) {
                    if constexpr (split_reader_cb_shared) {
                        wait_reserve_done(act_split_reader_reserve_done_semaphore_addr_ptr);
                        prev_addr = l1_write_addr_act;
                    } else {
                        l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                    }
                    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                    read_activation_data<
                        sliced_inner_dim,
                        dilation_w,
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        stride_w_bytes,
                        weight_size_w,
                        stride_w,
                        weight_size_h,
                        window_outer_offset>(
                        packed_reader_indices_ptr,
                        reader_offset,
                        l1_write_addr_act,
                        reader_idx,
                        act_l1_read_addr,
                        stride_h_bytes);
                    if constexpr (split_reader_cb_shared) {
                        // in case of shared cb we update the write address (it will remain the same if double buffering
                        // is not enabled)
                        l1_write_addr_act = split_reader_cb_write_addr_sum - prev_addr;
                        signal_write_done(act_split_reader_write_done_semaphore_addr_ptr);
                    }
                }
                if constexpr (!split_reader_cb_shared) {
                    cb_push_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                } else {
                    act_mcast_offset = act_mcast_offset_sum - act_mcast_offset;
                }
#endif
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
#if defined(SPLIT_READER) && !defined(SKIP_ACT_MCAST)
                    if (weight_tile_h_outer_i == act_mcast_sender_id) {
                        uint64_t act_multicast_data_addr =
                            act_multicast_noc_addr | (get_write_ptr(cb_id_act) + act_mcast_offset);
                        uint32_t self_write_start_address = get_write_ptr(cb_id_act) + act_mcast_offset;
                        noc_semaphore_wait(
                            act_mcast_sender_semaphore_addr_ptr_second,
                            act_mcast_num_dests + (is_receiver_core_act ? 0 : 1));
                        noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr_second, 0);

                        noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr_second, INVALID);
                        // Wait for first reader to signal that tilized data is ready
                        noc_semaphore_wait(act_first_reader_tilize_done_semaphore_addr_ptr, VALID);
                        noc_semaphore_set(act_first_reader_tilize_done_semaphore_addr_ptr, INVALID);

                        // Multicast the second half of activation data
                        if (is_receiver_core_act) {
                            if constexpr (act_mcast_num_cores) {
                                noc_async_write_multicast_loopback_src(
                                    tilized_act_start_address,
                                    act_multicast_data_addr,
                                    act_mcast_sender_size_bytes_second,
                                    act_mcast_num_cores + 1,
                                    true);
                            } else {
                                noc_async_write(
                                    get_noc_addr(tilized_act_start_address),
                                    get_noc_addr(self_write_start_address),
                                    act_mcast_sender_size_bytes_second);
                                noc_async_write_barrier();
                            }
                        } else {
                            noc_async_write_multicast(
                                tilized_act_start_address,
                                act_multicast_data_addr,
                                act_mcast_sender_size_bytes_second,
                                act_mcast_num_cores + 1,
                                true);
                        }

#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif

                        // Multicast VALID flag to receivers
                        if (is_receiver_core_act) {
                            if constexpr (act_mcast_num_cores) {
                                noc_semaphore_set_multicast_loopback_src(
                                    act_mcast_sender_semaphore_valid_addr,
                                    act_mcast_receiver_semaphore_noc_addr_second,
                                    act_mcast_num_cores + 1);
                                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr_second, VALID);
                            }
                        } else {
                            noc_semaphore_set_multicast(
                                act_mcast_sender_semaphore_valid_addr,
                                act_mcast_receiver_semaphore_noc_addr_second,
                                act_mcast_num_cores + 1);
                        }

                        // Signal to first reader that multicasting is done
                        noc_semaphore_set(act_second_reader_mcast_done_semaphore_addr_ptr, VALID);

                    } else if (is_receiver_core_act) {
                        // Receiver logic
                        noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr_second, INVALID);

                        // Atomic increment sender core counter
                        uint64_t act_mcast_sender_semaphore_noc_addr_second;
                        if constexpr (transpose_mcast) {
                            act_mcast_sender_semaphore_noc_addr_second = get_noc_addr(
                                act_mcast_sender_noc_x,
                                act_mcast_sender_noc_y[weight_tile_h_outer_i],
                                act_mcast_sender_semaphore_addr_second);
                        } else {
                            act_mcast_sender_semaphore_noc_addr_second = get_noc_addr(
                                act_mcast_sender_noc_y[weight_tile_h_outer_i],
                                act_mcast_sender_noc_x,
                                act_mcast_sender_semaphore_addr_second);
                        }
                        noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr_second, 1);

                        // Wait for VALID from sender
                        noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr_second, VALID);
                    }
                    act_mcast_offset = act_mcast_offset_sum - act_mcast_offset;
#endif  // SKIP_ACT_MCAST

                    // MCAST RECEIVE WEIGHTS
                    // read weight blocks inner dim
                    // read weight slice - 1 block of weights in width dim and full weight matrix height
                    // read slice only once for all activation blocks
                    cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                    // Set weights semaphore value to INVALID
                    noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // Atomic increment source core counter
                    noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                    // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                    noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

                    cb_push_back(cb_id_weight, weight_block_num_tiles);
                }  // for weight_block_height_num_outer
            }
#ifdef SPLIT_READER
            // Update reader index for next iteration (split reader increment)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
#endif
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
