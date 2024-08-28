// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    // in0 mcast args
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(2);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(3);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(4);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(5);

    // padding args
    const uint32_t last_block_h = get_arg_val<uint32_t>(6);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;

    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_block_stride = get_compile_time_arg_val(3);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    // in0 mcast args
    constexpr uint32_t in0_mcast_sender_semaphore_addr = get_compile_time_arg_val(8);
    constexpr uint32_t in0_mcast_receiver_semaphore_addr = get_compile_time_arg_val(9);
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(10);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(11);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(12);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(13);

    const bool transpose_core = (bool)get_arg_val<uint32_t>(7);
    const uint32_t noc_same_coord = get_arg_val<uint32_t>(8);
    tt_l1_ptr uint32_t* noc_diff_coord = (tt_l1_ptr uint32_t*)(get_arg_addr(9));

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

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

    cb_reserve_back(cb_id_in2, in0_block_num_tiles);
    uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
    uint32_t l1_write_addr_in2_step = in0_block_num_tiles * in0_single_tile_size_bytes;

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            // Operand 0
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint64_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
            uint32_t in0_block_size_bytes = 0;               // can be optimized later, pass it to kernel
            uint32_t sharded_addr_in2 = l1_write_addr_in2;
            uint64_t remote_sharded_addr;
            if (transpose_core) {
                remote_sharded_addr = get_noc_addr(noc_diff_coord[block], noc_same_coord, l1_write_addr_in2);
            } else {
                remote_sharded_addr = get_noc_addr(noc_same_coord, noc_diff_coord[block], l1_write_addr_in2);
            }

            // Copy in0 block into CB, as the default kernel
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
                    if (h < last_block_h) {
                        noc_async_read(remote_sharded_addr, l1_write_addr_in0, in0_single_tile_size_bytes);
                    }
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                    in0_block_size_bytes += in0_single_tile_size_bytes;
                    remote_sharded_addr += in0_single_tile_size_bytes;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();

#ifndef SKIP_MCAST
            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its value
            // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
            noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
            noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;

            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                in0_start_address, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores, true, true);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same
            // cmd_buf Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);
#endif

            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
        in0_tensor_start_tile_id += MtKt;
        l1_write_addr_in2 += l1_write_addr_in2_step;
    }
}
