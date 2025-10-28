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

    constexpr bool fuse_bias = get_compile_time_arg_val(18);

#ifdef SPLIT_READER
    constexpr bool split_reader_enabled = true;
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t window_outer = get_compile_time_arg_val(6);  // num_blocks_act_w
    constexpr bool sliced_inner_dim = window_outer > 1;             // Derived like block sharded reader
    constexpr uint32_t act_block_num_tiles_split_last = get_compile_time_arg_val(19);  // This is what factory passes
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(21);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(22);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(23);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(25);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(26);
    constexpr uint32_t stride_w = get_compile_time_arg_val(27);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(28);  // Input filter window height

    // When the split reader CB is shared, both readers write to the same circular buffer.
    // Synchronization is required: the main reader signals when CB space is reserved,
    // and the second reader signals when it has finished writing its portion.
    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(29) == 1;
    const uint32_t act_split_reader_reserve_done_semaphore_addr =
        (split_reader_cb_shared) ? get_semaphore(get_compile_time_arg_val(30)) : 0;
    const uint32_t act_split_reader_write_done_semaphore_addr =
        (split_reader_cb_shared) ? get_semaphore(get_compile_time_arg_val(31)) : 0;
    constexpr uint32_t act_write_offset = get_compile_time_arg_val(32);
    constexpr uint32_t act_write_offset_last = get_compile_time_arg_val(33);

    volatile tt_l1_ptr uint32_t* act_split_reader_reserve_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_reserve_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_split_reader_write_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_write_done_semaphore_addr);

    const uint32_t split_reader_cb_write_addr = get_write_ptr(cb_id_act_second_reader) + act_write_offset;
    // In case of double buffering the split reader can write to two different addresses
    const uint32_t split_reader_cb_write_addr_last = get_write_ptr(cb_id_act_second_reader) + act_write_offset_last;
    const uint32_t split_reader_cb_write_addr_sum = split_reader_cb_write_addr + split_reader_cb_write_addr_last;

    constexpr bool transpose_mcast = get_compile_time_arg_val(34) == 1;
    const uint32_t act_mcast_reserve_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(35));
    const uint32_t act_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(36));
    constexpr uint32_t act_cb_id = get_compile_time_arg_val(37);
    constexpr uint32_t act_tilized_cb = get_compile_time_arg_val(38);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(39);
    // act_mcast_write_offset represents the offset to the second half of the ACT CB (which will be written to by the
    // second reader)
    constexpr uint32_t act_mcast_write_offset = get_compile_time_arg_val(40);
    // act_mcast_write_offset_last represents the offset to the second half of the second block of ACT CB (which will be
    // written to by the main reader, when double buffering is enabled)
    constexpr uint32_t act_mcast_write_offset_last = get_compile_time_arg_val(41);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(42);
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(43);
    constexpr bool skip_mcast = get_compile_time_arg_val(44) == 1;

    volatile tt_l1_ptr uint32_t* act_mcast_reserve_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_reserve_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);

    // L1 array
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);

    constexpr uint32_t act_tilized_offset =
        act_mcast_write_offset;  // This is the offset to the first half of the ACT_TILIZED CB (which will be read by
                                 // the second reader)
    const uint32_t tilized_act_start_address = get_read_ptr(act_tilized_cb) + act_tilized_offset;
    const uint32_t base_act_address = get_write_ptr(act_cb_id);
#else
    constexpr bool split_reader_enabled = false;
    const uint32_t split_reader_cb_write_addr = 0;
    volatile tt_l1_ptr uint32_t* act_mcast_reserve_done_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = nullptr;
    const uint32_t act_mcast_receiver_semaphore_addr = 0;
    constexpr uint32_t act_cb_id = 0;
    constexpr uint32_t act_tilized_cb = 0;
    constexpr uint32_t cb_l1_array = 0;
    constexpr uint32_t act_mcast_write_offset = 0;
    constexpr uint32_t act_mcast_write_offset_last = 0;
    constexpr uint32_t act_tilized_offset = 0;
    constexpr uint32_t act_write_offset = 0;
    constexpr uint32_t act_write_offset_last = 0;
    constexpr uint32_t act_mcast_num_cores = 0;
    constexpr uint32_t act_mcast_sender_size_bytes = 0;
    const uint32_t tilized_act_start_address = 0;
    const uint32_t base_act_address = 0;
    uint32_t act_mcast_sender_semaphore_valid_addr = 0;
    constexpr bool skip_mcast = true;
#endif
    constexpr uint64_t act_mcast_write_offset_sum = act_mcast_write_offset + act_mcast_write_offset_last;

    // mcast args
    uint32_t i = 0;
    const uint32_t weights_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_receiver_core = get_arg_val<uint32_t>(i++) > 0;

    uint32_t act_mcast_dest_noc_start_x = 0;
    uint32_t act_mcast_dest_noc_start_y = 0;
    uint32_t act_mcast_dest_noc_end_x = 0;
    uint32_t act_mcast_dest_noc_end_y = 0;
    uint32_t act_mcast_sender_id = 0;
    if constexpr (split_reader_enabled) {
        act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
        act_mcast_sender_id = get_arg_val<uint32_t>(i++);
    }
    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x, act_mcast_dest_noc_start_y, act_mcast_dest_noc_end_x, act_mcast_dest_noc_end_y, 0);
    uint64_t act_multicast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;
    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    const uint64_t weights_mcast_sender_semaphore_noc_addr =
        get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);

#ifdef SPLIT_READER
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

    uint32_t cb_ind_offset = 0;
#endif
    // read in bias if enabled (done only once for all batches)
    bool load_bias = true;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t l1_write_addr_act = split_reader_cb_write_addr;
    uint32_t prev_addr = 0;
    uint64_t act_write_offset_current = act_mcast_write_offset;

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
                    // When double buffering is enabled, we need to update the write offset to the second half of the
                    // ACT CB
                    act_write_offset_current = act_mcast_write_offset_sum - act_write_offset_current;
                }
#endif
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    if constexpr (split_reader_enabled && !skip_mcast) {
                        if (weight_tile_h_outer_i == act_mcast_sender_id) {
                            uint64_t act_address = base_act_address + act_write_offset_current;
                            uint64_t act_multicast_data_addr = act_multicast_noc_addr | act_address;

                            wait_reserve_done(act_mcast_reserve_done_semaphore_addr_ptr);

                            if (is_receiver_core) {
                                if constexpr (act_mcast_num_cores) {
                                    // num_dests will source, since we are copying to a different local CB as well
                                    noc_async_write_multicast_loopback_src(
                                        tilized_act_start_address,
                                        act_multicast_data_addr,
                                        act_mcast_sender_size_bytes,
                                        act_mcast_num_cores + 1,
                                        true);
                                } else {
                                    // In this case sender core is the only reciever in the grid,
                                    // we can't use the multicast_loopback_src (hang)
                                    noc_async_write(
                                        get_noc_addr(tilized_act_start_address),
                                        get_noc_addr(act_address),
                                        act_mcast_sender_size_bytes);
                                    noc_async_write_barrier();
                                }
                            } else {
                                // If sender core is not the reciever core as well we can't use the loopback mcast.
                                // (hang)
                                noc_async_write_multicast(
                                    tilized_act_start_address,
                                    act_multicast_data_addr,
                                    act_mcast_sender_size_bytes,
                                    act_mcast_num_cores + 1,
                                    true);
                            }

                            // Note: no need for write barrier, since these two multicasts are done on the same noc id
                            // and same vc even though cmd bufs are different Also, this only works because we are
                            // setting VCs statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                            // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs
                            // and may not be sent in order they are issued
                            noc_async_writes_flushed();
#endif

                            if (is_receiver_core) {
                                // We should also multicast VALID flag to destinations for receiver semaphore
                                if constexpr (act_mcast_num_cores) {
                                    noc_semaphore_set_multicast_loopback_src(
                                        act_mcast_sender_semaphore_valid_addr,
                                        act_multicast_receiver_semaphore_noc_addr,
                                        act_mcast_num_cores + 1);
                                    noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
                                }
                            } else {
                                noc_semaphore_set_multicast(
                                    act_mcast_sender_semaphore_valid_addr,
                                    act_multicast_receiver_semaphore_noc_addr,
                                    act_mcast_num_cores + 1);
                            }
                        }
                        act_write_offset_current = act_mcast_write_offset_sum - act_write_offset_current;
                    }
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

        }  // out_num_blocks_h
    }  // out_num_blocks_w

    noc_async_write_barrier();
}
