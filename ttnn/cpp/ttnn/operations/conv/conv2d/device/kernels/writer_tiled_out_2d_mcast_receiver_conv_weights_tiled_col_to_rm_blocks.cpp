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

#ifdef SPLIT_READER_OVERLAPPED
    const uint32_t act_split_reader_sync_first_semaphore_addr = get_semaphore(get_compile_time_arg_val(28));
    const uint32_t act_split_reader_sync_second_semaphore_addr = get_semaphore(get_compile_time_arg_val(29));
    constexpr uint32_t act_write_offset = get_compile_time_arg_val(30);
    constexpr uint32_t act_block_size = get_compile_time_arg_val(31);
    constexpr uint32_t read_ind_stride = get_compile_time_arg_val(32);
    constexpr uint32_t act_cb_block_cnt = get_compile_time_arg_val(33);

    const uint32_t base_write_addr = get_write_ptr(cb_id_act_second_reader);

    volatile tt_l1_ptr uint32_t* act_split_reader_sync_first_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_sync_first_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_split_reader_sync_second_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_sync_second_semaphore_addr);
#endif
#endif

    // mcast args
    uint32_t i = 0;
    const uint32_t weights_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;

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
#ifdef FUSE_BIAS
    bool load_bias = true;
#endif

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
#ifdef SPLIT_READER
            // Read activation data using block sharded pattern (for second reader)
            uint32_t reader_offset = act_l1_read_addr;
#endif
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
#ifdef SPLIT_READER
                reader_idx = start_reader_idx;
#ifndef SPLIT_READER_OVERLAPPED
                cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
#endif

                if (is_sender_core) {
#ifdef SPLIT_READER_OVERLAPPED
                    noc_semaphore_wait(act_split_reader_sync_first_semaphore_addr_ptr, VALID);
                    noc_semaphore_set(act_split_reader_sync_first_semaphore_addr_ptr, INVALID);
                    uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader) + act_write_offset;
#else
                    uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
#endif
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
#ifdef SPLIT_READER_OVERLAPPED
                    cb_ind_offset = (cb_ind_offset + read_ind_stride) % act_cb_block_cnt;
                    get_local_cb_interface(cb_id_act_second_reader).fifo_wr_ptr =
                        base_write_addr + cb_ind_offset * act_block_size;
                    noc_semaphore_set(act_split_reader_sync_second_semaphore_addr_ptr, VALID);
#endif
                }
#ifndef SPLIT_READER_OVERLAPPED
                cb_push_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
#endif
#endif
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
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
