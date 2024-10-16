// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(4);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(5);
    // block args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(6);
    // in0 mcast args
    constexpr uint32_t in0_mcast_dest_noc_start_x = get_compile_time_arg_val(7);
    constexpr uint32_t in0_mcast_dest_noc_start_y = get_compile_time_arg_val(8);
    constexpr uint32_t in0_mcast_dest_noc_end_x = get_compile_time_arg_val(9);
    constexpr uint32_t in0_mcast_dest_noc_end_y = get_compile_time_arg_val(10);
    // in0 semaphore always valid
    uint32_t in0_mcast_sender_valid_semaphore = get_semaphore(get_compile_time_arg_val(11));

    constexpr uint32_t num_blocks_per_shard = get_compile_time_arg_val(12);
    constexpr uint32_t num_storage_cores = num_blocks / num_blocks_per_shard;

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg_val<uint32_t>(0);
    // if not worker core, skip
    if (worker_core_type == 0) {
        return;
    }
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + num_storage_cores));

    const uint32_t sender_block_id = sender_id * num_blocks_per_shard;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(in0_mcast_dest_noc_start_x,
                                                                                  in0_mcast_dest_noc_start_y,
                                                                                  in0_mcast_dest_noc_end_x,
                                                                                  in0_mcast_dest_noc_end_y,
                                                                                  in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

    uint32_t local_read_addr = get_read_ptr(cb_id_in2);

    if (worker_core_type == 1) {  // mcast sender + no compute

        for (uint32_t i = 0; i < num_blocks_per_shard; ++i) {
            const uint32_t block_id = sender_block_id + i;

            // Operand 0
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            if (block_id % 2 != 0) {  // double buffer
                l1_write_addr_in0 += in0_block_size_bytes;
            }

            uint64_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting

            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
            noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
            noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;

#ifndef SKIP_MCAST
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                local_read_addr, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores, true, true);
#endif

            noc_semaphore_set_multicast(
                in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);

            local_read_addr += in0_block_size_bytes;
        }

    } else if (worker_core_type == 2) {  // mcast sender + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = block / num_blocks_per_shard;

            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            // Set in0 semaphore value to INVALID
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            if (block_id == sender_id) {
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                uint64_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting

                // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests - 1);
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;
#ifndef SKIP_MCAST
                noc_async_write_multicast_loopback_src(
                    local_read_addr, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores, true, true);
#endif
                noc_semaphore_set_multicast_loopback_src(
                    in0_mcast_sender_valid_semaphore, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);

                local_read_addr += in0_block_size_bytes;

            } else {
                uint64_t in0_mcast_sender_semaphore_noc_addr = get_noc_addr(in0_mcast_sender_noc_x[block_id],
                                                                            in0_mcast_sender_noc_y[block_id],
                                                                            in0_mcast_sender_semaphore_addr);

                // Atomic increment source core counter
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            }

            noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
    } else {  // mcast receiver + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = block / num_blocks_per_shard;

            // get the mcast sender noc
            uint64_t in0_mcast_sender_semaphore_noc_addr = get_noc_addr(
                in0_mcast_sender_noc_x[block_id], in0_mcast_sender_noc_y[block_id], in0_mcast_sender_semaphore_addr);

            // Operand 0
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
    }
}
