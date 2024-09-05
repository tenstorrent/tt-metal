// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

//Only a part of the total channel depth (width) is used in one block.
template<int window_height, int window_width>
FORCE_INLINE void read_channels(uint32_t& l1_write_addr_act, const uint32_t act_l1_read_addr, const uint32_t reader_channel_idx,
        const uint32_t conv_act_c_bytes, const uint32_t conv_act_c_read_bytes, const uint32_t stride_h_bytes, const uint32_t stride_w_bytes) {

    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
    #pragma GCC unroll 3
    for(uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
        for (uint32_t inner = 0; inner < window_width; inner++) {
            //Read the partial depth.
            noc_async_read_one_packet_with_state<true>(act_l1_read_addr_row_offset, l1_write_addr_act);
            //Increment by full depth to go to the next pixel
            l1_write_addr_act += conv_act_c_read_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        //Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {

    constexpr bool act_in_dram                          = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h                         = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w                         = get_compile_time_arg_val(2);
    constexpr uint32_t dilation_h                       = get_compile_time_arg_val(3);
    constexpr uint32_t dilation_w                       = get_compile_time_arg_val(4);
    constexpr uint32_t conv_act_size_w                  = get_compile_time_arg_val(5);
    constexpr uint32_t conv_act_c_read_bytes            = get_compile_time_arg_val(6);
    constexpr uint32_t weight_size_h                    = get_compile_time_arg_val(7);
    constexpr uint32_t weight_size_w                    = get_compile_time_arg_val(8);
    constexpr uint32_t act_block_h_datums               = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_num_tiles              = get_compile_time_arg_val(10);
    constexpr uint32_t act_w_num_outer                  = get_compile_time_arg_val(11);
    constexpr uint32_t act_num_blocks_w                 = get_compile_time_arg_val(12);
    const uint32_t act_mcast_sender_semaphore_addr      = get_semaphore(get_compile_time_arg_val(13));
    const uint32_t act_mcast_receiver_semaphore_addr    = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t act_mcast_dest_noc_start_x       = get_compile_time_arg_val(15);
    constexpr uint32_t act_mcast_dest_noc_start_y       = get_compile_time_arg_val(16);
    constexpr uint32_t act_mcast_dest_noc_end_x         = get_compile_time_arg_val(17);
    constexpr uint32_t act_mcast_dest_noc_end_y         = get_compile_time_arg_val(18);
    constexpr uint32_t act_mcast_sender_size_bytes      = get_compile_time_arg_val(19);
    constexpr uint32_t act_mcast_num_cores              = get_compile_time_arg_val(20);


    uint32_t i = 0; //Runtime arg index

    uint32_t this_core_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t this_core_y = get_arg_val<uint32_t>(i); i+=1;

    //Num of cols of compute cores. (Total Cores, not active cores.)
    uint32_t num_cores_x = get_arg_val<uint32_t>(i); i+=1;

    //X and Y lookup are independant.
    //X Lookup table for translating logical to physical cores.
    tt_l1_ptr uint32_t *act_mcast_x_lookup  = (tt_l1_ptr uint32_t*)(get_arg_addr(i));
    i+=num_cores_x;
    //Y lookup.
    tt_l1_ptr uint32_t *act_mcast_y_lookup  = (tt_l1_ptr uint32_t*)(get_arg_addr(i));


    // DPRINT<<"Act Params L1 :  "<<dilation_h<<" "<<dilation_w<<" "<<conv_act_size_w<<"  "<<conv_act_c_read_bytes<<"  "<<weight_size_h<<"  "<<weight_size_w<<"  "<<act_block_h_datums<<"  "<<act_block_num_tiles<<ENDL()<<
    // "L2  "<<act_w_num_outer<<"  "<<act_num_blocks_w<<"  "<<act_mcast_sender_semaphore_addr<<"  "<<act_mcast_receiver_semaphore_addr<<"  "<<act_mcast_dest_noc_start_x<<
    // "L3  "<<act_mcast_dest_noc_start_y<<"  "<<act_mcast_dest_noc_end_x<<"  "<<act_mcast_dest_noc_end_y<<"  "<<act_mcast_sender_size_bytes<<"  "<<act_mcast_num_cores<<ENDL();

    //Equivalent to Core Index.
    uint32_t act_mcast_sender_id = this_core_x + (num_cores_x * this_core_y) ;

    constexpr uint32_t cb_id_act = tt::CB::c_in0;
    constexpr uint32_t cb_id_weight = tt::CB::c_in1;

    constexpr uint32_t tilized_in0_cb_id = tt::CB::c_intermed1;
    constexpr uint32_t cb_id_sharded_act = tt::CB::c_in3;
    constexpr uint32_t cb_id_act_row_major_bfloat16 = tt::CB::c_in6;

    constexpr uint32_t cb_reader_indices = tt::CB::c_in4;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // L1 array
    constexpr uint32_t cb_l1_array = tt::CB::c_in5;
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));

    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    act_mcast_sender_semaphore_valid_addr_ptr[0] = 1; // Load const 1 to be used as semaphore valid value sent from sender to receivers

    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(act_mcast_sender_semaphore_valid_addr_ptr);

    // Set up remote VALID value
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, VALID);

    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr);

    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x,
        act_mcast_dest_noc_start_y,
        act_mcast_dest_noc_end_x,
        act_mcast_dest_noc_end_y,
        0
    );

    uint64_t act_mcast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;

    //Compute is divided along the width to reduce the size of CBs.
    //Only a part of the width on each core is used in one block.
    //Bytes read is conv_act_c_read_bytes.
    //Size of channel in bytes on this core is conv_act_c_bytes.
    constexpr uint32_t conv_act_c_bytes = conv_act_c_read_bytes * act_num_blocks_w;

    //Stride after each channel read.
    constexpr uint32_t stride_w_bytes = conv_act_c_bytes * dilation_w;

    //Striding to next row happens using stride_h_bytes
    constexpr uint32_t stride_h_bytes = (conv_act_size_w ) * conv_act_c_bytes * dilation_h;

    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), conv_act_c_read_bytes);

    // Reset reader_idx to finish act_block_h_datums

    for(uint32_t block_w_index = 0; block_w_index < act_num_blocks_w; block_w_index++)
    {
        uint32_t reader_idx = 0;
        cb_reserve_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);
        uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);
        // DPRINT<<"L1 Write Addr "<<l1_write_addr_act<<"\n";


        for (uint32_t bh = 0; bh < act_block_h_datums / 2; bh++) {
            uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
            read_channels<weight_size_h,weight_size_w>(l1_write_addr_act, act_l1_read_addr, two_reader_indices & 0xffff, conv_act_c_bytes, conv_act_c_read_bytes, stride_h_bytes, stride_w_bytes);
            read_channels<weight_size_h,weight_size_w>(l1_write_addr_act, act_l1_read_addr, two_reader_indices >> 16   , conv_act_c_bytes, conv_act_c_read_bytes, stride_h_bytes, stride_w_bytes);

            reader_idx++;
        }

        //After reading one block, increment the starting read pointer by the width of the block.
        //Next read uses the next set of channels.
        act_l1_read_addr +=conv_act_c_read_bytes;

        noc_async_read_barrier();
        cb_push_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);


        // Round robin self-mcast and receive tilized act matrix in cb_id_act
        // Compute should function like regular mm

        uint32_t act_w_outer_i = 0;

        uint32_t sender_noc_x = 0;
        uint32_t sender_noc_y = 0;

        for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            if (act_w_outer_i == act_mcast_sender_id) {
                // MCAST SENDER: send entire tilized input to other cores in column
                // wait until all act mcast destinations have atomically incremented the act semaphore_addr (i.e. its value should be act_mcast_num_dests), then reset
                // the semaphore_addr value back to zero for the next block

                noc_semaphore_wait_min(act_mcast_sender_semaphore_addr_ptr, act_mcast_num_cores-1);
                noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                // // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                cb_wait_front(tilized_in0_cb_id, act_block_num_tiles);

                // // Now we have the block in the CB address, we can mcast to dests!
                uint32_t tilized_act_start_address = get_read_ptr(tilized_in0_cb_id);

                uint64_t act_multicast_data_addr = act_multicast_noc_addr | get_write_ptr(cb_id_act);
                // // num_dests will source, since we are copying to a different local CB as well
                noc_async_write_multicast_loopback_src(tilized_act_start_address, act_multicast_data_addr, act_mcast_sender_size_bytes, act_mcast_num_cores , false, false);


                // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
                // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

                // // We should also multicast VALID flag to destinations for receiver semaphore
                noc_semaphore_set_multicast_loopback_src(act_mcast_sender_semaphore_valid_addr, act_mcast_receiver_semaphore_noc_addr, act_mcast_num_cores , false, false);


                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);

            } else {

                // MCAST RECEIVER: receive entire tilized input from sender core
                // Set act semaphore value to INVALID
                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                uint32_t sender_x = act_mcast_x_lookup[sender_noc_x];
                uint32_t sender_y = act_mcast_y_lookup[sender_noc_y];

                // Atomic increment source core counter
                uint64_t act_mcast_sender_semaphore_noc_addr = get_noc_addr(sender_x, sender_y, act_mcast_sender_semaphore_addr);
                noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

                // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)

                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
            }
            cb_push_back(cb_id_act, act_block_num_tiles);

            sender_noc_x++;
            if(sender_noc_x >= num_cores_x) {
                sender_noc_x = 0;
                sender_noc_y++;
            }

        } // act_w_num_outer
        cb_pop_front(tilized_in0_cb_id, act_block_num_tiles);
    }
}
