// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

constexpr uint32_t weight_size_h = get_compile_time_arg_val(5);
constexpr uint32_t weight_size_w = get_compile_time_arg_val(6);
// Only a part of the total channel depth (width) is used in one block.
template <int window_height, int window_width>
FORCE_INLINE void read_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t stride_h_bytes,
    const uint32_t stride_w_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
#pragma GCC unroll weight_size_h
    for (uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
#pragma GCC unroll weight_size_w
        for (uint32_t inner = 0; inner < window_width; inner++) {
            // Read the partial depth.
            noc_async_read_one_packet_with_state<true>(act_l1_read_addr_row_offset, l1_write_addr_act);
            // Increment by full depth to go to the next pixel
            l1_write_addr_act += conv_act_c_read_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        // Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {
    constexpr uint32_t stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(1);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w = get_compile_time_arg_val(3);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(7);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_input_cores = get_compile_time_arg_val(9);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_w = get_compile_time_arg_val(11);
    const uint32_t act_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    const uint32_t act_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(13));
    constexpr uint32_t act_mcast_dest_noc_start_x = get_compile_time_arg_val(14);
    constexpr uint32_t act_mcast_dest_noc_start_y = get_compile_time_arg_val(15);
    constexpr uint32_t act_mcast_dest_noc_end_x = get_compile_time_arg_val(16);
    constexpr uint32_t act_mcast_dest_noc_end_y = get_compile_time_arg_val(17);
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(19);
    constexpr uint32_t num_reader_cores = get_compile_time_arg_val(20);

    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(24);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(25);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(26);

    constexpr uint32_t num_mcast_cores = num_input_cores > num_output_cores ? num_input_cores : num_output_cores;
    uint32_t i = 0;  // Runtime arg index

    uint32_t this_core_x = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t this_core_y = get_arg_val<uint32_t>(i);
    i += 1;

    // Num of cols of compute cores. (Total Cores, not active cores.)
    uint32_t num_cores_x = get_arg_val<uint32_t>(i);
    i += 1;

    // X and Y lookup are independant.
    // X Lookup table for translating logical to physical cores.
    tt_l1_ptr uint32_t* act_mcast_x_lookup = (tt_l1_ptr uint32_t*)(get_arg_addr(i));
    i += num_cores_x;
    // Y lookup.
    tt_l1_ptr uint32_t* act_mcast_y_lookup = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

    // Equivalent to Core Index.
    uint32_t this_core_id = this_core_x + (num_cores_x * this_core_y);

    if (this_core_id >= num_mcast_cores) {
        return;
    }
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // L1 array
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));

    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    act_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers

    uint32_t act_mcast_sender_semaphore_valid_addr =
        reinterpret_cast<uint32_t>(act_mcast_sender_semaphore_valid_addr_ptr);

    // Set up remote VALID value
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, VALID);

    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr);

    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x, act_mcast_dest_noc_start_y, act_mcast_dest_noc_end_x, act_mcast_dest_noc_end_y, 0);

    uint64_t act_mcast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;

    // Compute is divided along the width to reduce the size of CBs.
    // Only a part of the width on each core is used in one block.
    // Bytes read is conv_act_c_read_bytes.
    // Size of channel in bytes on this core is conv_act_c_bytes.
    constexpr uint32_t conv_act_c_bytes = conv_act_c_read_bytes * act_num_blocks_w;

    // Stride after each channel read.
    constexpr uint32_t stride_w_bytes = conv_act_c_bytes * dilation_w;

    // Striding to next row happens using stride_h_bytes
    constexpr uint32_t stride_h_bytes = (conv_act_size_w)*conv_act_c_bytes * dilation_h;

    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), conv_act_c_read_bytes);
    uint32_t reader_idx = 0;
    uint32_t l1_write_addr_act = 0;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ntile_height = act_block_h_datums / TILE_HEIGHT;
    constexpr uint32_t ntile_width = act_block_num_tiles / ntile_height;

    // Reset reader_idx to finish act_block_h_datums
    for (uint32_t block_h_index = 0; block_h_index < act_num_blocks_h; block_h_index++) {
        act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
        uint32_t old_reader_idx = reader_idx;
        for (uint32_t block_w_index = 0; block_w_index < act_num_blocks_w; block_w_index++) {
            reader_idx = old_reader_idx;
            if (this_core_id < num_input_cores) {
                uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
                uint16_t num_elems = two_reader_indices & 0xffff;

                uint16_t remaining_indexes = TILE_HEIGHT;
                while (num_elems--) {
                    reader_idx++;
                    two_reader_indices = packed_reader_indices_ptr[reader_idx];
                    uint16_t start_ind = two_reader_indices & 0xffff;
                    uint16_t end_ind = two_reader_indices >> 16;
                    for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                        if (remaining_indexes == TILE_HEIGHT) {
                            l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);
                            cb_reserve_back(cb_id_act_row_major_bfloat16, ntile_width);
                        }
                        read_channels<weight_size_h, weight_size_w>(
                            l1_write_addr_act,
                            act_l1_read_addr,
                            ind,
                            conv_act_c_bytes,
                            conv_act_c_read_bytes,
                            stride_h_bytes,
                            stride_w_bytes);

                        if (--remaining_indexes == 0) {
                            noc_async_read_barrier();
                            cb_push_back(cb_id_act_row_major_bfloat16, ntile_width);
                            l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);
                            remaining_indexes = TILE_HEIGHT;
                        }
                    }
                }
                if (remaining_indexes && remaining_indexes != TILE_HEIGHT) {
                    noc_async_read_barrier();
                    cb_push_back(cb_id_act_row_major_bfloat16, ntile_width);
                }
                reader_idx++;

                // After reading one block, increment the starting read pointer by the width of the block.
                // Next read uses the next set of channels.
                act_l1_read_addr += conv_act_c_read_bytes;
            } else {
                for (uint32_t tile_h_index = 0; tile_h_index < ntile_height; tile_h_index++) {
                    cb_reserve_back(cb_id_act_row_major_bfloat16, ntile_width);
                    cb_push_back(cb_id_act_row_major_bfloat16, ntile_width);
                }
            }

            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm

            uint32_t act_w_outer_i = 0;

            uint32_t sender_noc_x = 0;
            uint32_t sender_noc_y = 0;

            for (uint32_t act_w_outer_i = 0; act_w_outer_i < num_input_cores; act_w_outer_i++) {
                cb_reserve_back(cb_id_act, act_block_num_tiles);
                if (act_w_outer_i == this_core_id) {
                    // MCAST SENDER: send entire tilized input to other cores in column
                    // wait until all act mcast destinations have atomically incremented the act semaphore_addr (i.e.
                    // its value should be act_mcast_num_dests), then reset the semaphore_addr value back to zero for
                    // the next block

                    noc_semaphore_wait_min(act_mcast_sender_semaphore_addr_ptr, num_mcast_cores - 1);
                    noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

                    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                    cb_wait_front(tilized_in0_cb_id, act_block_num_tiles);

                    // Now we have the block in the CB address, we can mcast to dests!
                    uint32_t tilized_act_start_address = get_read_ptr(tilized_in0_cb_id);

                    // num_dests will source, since we are copying to a different local CB as well
                    uint64_t act_multicast_data_addr = act_multicast_noc_addr | get_write_ptr(cb_id_act);

                    noc_async_write_multicast_loopback_src(
                        tilized_act_start_address,
                        act_multicast_data_addr,
                        act_mcast_sender_size_bytes,
                        num_reader_cores,
                        true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and same
                    // vc even though cmd bufs are different Also, this only works because we are setting VCs statically
                    // (using NOC_CMD_STATIC_VC).

#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may
                    // not be sent in order they are issued
                    noc_async_writes_flushed();
#endif
                    // We should also multicast VALID flag to destinations for receiver semaphore
                    noc_semaphore_set_multicast_loopback_src(
                        act_mcast_sender_semaphore_valid_addr,
                        act_mcast_receiver_semaphore_noc_addr,
                        num_reader_cores,
                        false);
                    noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
                } else {
                    // MCAST RECEIVER: receive entire tilized input from sender core
                    // Set act semaphore value to INVALID
                    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                    uint32_t sender_x = act_mcast_x_lookup[sender_noc_x];
                    uint32_t sender_y = act_mcast_y_lookup[sender_noc_y];

                    // Atomic increment source core counter
                    uint64_t act_mcast_sender_semaphore_noc_addr =
                        get_noc_addr(sender_x, sender_y, act_mcast_sender_semaphore_addr);
                    noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

                    // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                    noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
                }

                cb_push_back(cb_id_act, act_block_num_tiles);

                sender_noc_x++;
                if (sender_noc_x >= num_cores_x) {
                    sender_noc_x = 0;
                    sender_noc_y++;
                }

            }  // num_input_cores
            cb_pop_front(tilized_in0_cb_id, act_block_num_tiles);
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
}
