// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

constexpr uint32_t weight_size_h = get_compile_time_arg_val(7);  // Input filter window height
constexpr uint32_t weight_size_w = get_compile_time_arg_val(8);  // Input filter window width

template <int window_height, int window_width>
FORCE_INLINE void read_dilated_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
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
            l1_write_addr_act += conv_act_c_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        // Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

FORCE_INLINE
void read_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t coalesced_read_bytes,
    const uint32_t stride_h_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_read_bytes);
#pragma GCC unroll weight_size_h
    for (uint32_t inner = 0; inner < weight_size_h; inner++) {
        noc_async_read_one_packet_with_state<true>(act_l1_read_addr_plus_offset, l1_write_addr_act);
        l1_write_addr_act += coalesced_read_bytes;
        // +2 is hard-coded, TODO: generalize
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

constexpr uint32_t DILATION_W = get_compile_time_arg_val(1);
void kernel_main() {
    constexpr uint32_t dilation_h = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t window_inner = get_compile_time_arg_val(5);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t act_w_num_outer = get_compile_time_arg_val(13);
    constexpr uint32_t act_mcast_num_dests = get_compile_time_arg_val(14);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(15);
    const uint32_t act_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    const uint32_t act_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(17));
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(18);
    constexpr bool transpose_mcast = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(25);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(26);

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i);
    i += 1;

    if (noop) {
        return;
    }

    uint32_t act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t act_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t act_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t act_mcast_sender_id = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t act_mcast_sender_noc_x = get_arg_val<uint32_t>(i);
    i += 1;

    tt_l1_ptr uint32_t* act_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // L1 array
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);
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

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // Fully create act matrix and tilize it before mcast
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

    // Reset reader_idx to finish act_block_h_datums
    uint32_t reader_idx = 0;
    for (uint32_t nbh = 0; nbh < act_num_blocks_h; nbh++) {
        cb_reserve_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);
        uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);

        constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
        constexpr uint32_t stride_w_bytes = conv_act_c_read_bytes * dilation_w;

        uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;
        while (num_elems--) {
            reader_idx++;
            uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
            uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                if constexpr (DILATION_W == 1) {
                    read_channels(
                        l1_write_addr_act,
                        act_l1_read_addr,
                        ind,
                        conv_act_c_read_bytes,
                        coalesced_read_bytes,
                        stride_h_bytes);
                    if constexpr (act_block_w_extra_align_bytes) {
                        l1_write_addr_act += act_block_w_extra_align_bytes;
                    }
                } else {
                    read_dilated_channels<weight_size_h, weight_size_w>(
                        l1_write_addr_act,
                        act_l1_read_addr,
                        ind,
                        conv_act_c_read_bytes,
                        stride_h_bytes,
                        stride_w_bytes);
                }
            }
        }
        reader_idx++;

        noc_async_read_barrier();
        cb_push_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);

        // Round robin self-mcast and receive tilized act matrix in cb_id_act
        // Compute should function like regular mm
        for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            if (act_w_outer_i == act_mcast_sender_id) {
                // MCAST SENDER: send entire tilized input to other cores in column
                // wait until all act mcast destinations have atomically incremented the act semaphore_addr (i.e. its
                // value should be act_mcast_num_dests), then reset the semaphore_addr value back to zero for the next
                // block
                noc_semaphore_wait(act_mcast_sender_semaphore_addr_ptr, act_mcast_num_dests);
                noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                cb_wait_front(tilized_in0_cb_id, act_block_num_tiles);

                // Now we have the block in the CB address, we can mcast to dests!
                uint32_t tilized_act_start_address = get_read_ptr(tilized_in0_cb_id);

                uint64_t act_multicast_data_addr = act_multicast_noc_addr | get_write_ptr(cb_id_act);
                // num_dests will source, since we are copying to a different local CB as well
                noc_async_write_multicast_loopback_src(
                    tilized_act_start_address,
                    act_multicast_data_addr,
                    act_mcast_sender_size_bytes,
                    act_mcast_num_cores + 1,
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
                // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
                // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not
                // be sent in order they are issued
                noc_async_writes_flushed();
#endif

                // We should also multicast VALID flag to destinations for receiver semaphore
                noc_semaphore_set_multicast_loopback_src(
                    act_mcast_sender_semaphore_valid_addr,
                    act_mcast_receiver_semaphore_noc_addr,
                    act_mcast_num_cores + 1);

                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
            } else {
                // MCAST RECEIVER: receive entire tilized input from sender core
                // Set act semaphore value to INVALID
                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                uint64_t act_mcast_sender_semaphore_noc_addr;
                if constexpr (transpose_mcast) {
                    act_mcast_sender_semaphore_noc_addr = get_noc_addr(
                        act_mcast_sender_noc_x, act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_semaphore_addr);
                } else {
                    act_mcast_sender_semaphore_noc_addr = get_noc_addr(
                        act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_noc_x, act_mcast_sender_semaphore_addr);
                }
                noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

                // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
            }
            cb_push_back(cb_id_act, act_block_num_tiles);
        }  // act_w_num_outer
        cb_pop_front(tilized_in0_cb_id, act_block_num_tiles);
    }
    noc_async_write_barrier();
}
