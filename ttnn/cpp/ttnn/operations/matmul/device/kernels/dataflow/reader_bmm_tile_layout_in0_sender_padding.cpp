// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

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
    constexpr bool extract_shard_sub_blocks = (bool)get_compile_time_arg_val(7);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(9);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(10);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(13);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(14);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(15);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(16);

#ifdef MATMUL_SIGNAL
    /* Overlapped all-gather with matmul params */
    constexpr uint32_t num_transfers = get_compile_time_arg_val(17);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr uint32_t start_ring_index = get_compile_time_arg_val(19);
    constexpr uint32_t tensor_slice_shape_width = get_compile_time_arg_val(20);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(21);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(22);
    constexpr uint32_t is_clockwise_direction = get_compile_time_arg_val(23);
    const uint32_t signal_op_sem_addr_dir0 = get_semaphore(get_compile_time_arg_val(24));
    const uint32_t signal_op_sem_addr_dir1 = get_semaphore(get_compile_time_arg_val(25));

    // Internal semaphores
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir0);
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir1);

    // Start idxs for the different directions
    uint32_t ring_index_dir0 = start_ring_index;
    // Adjust to include copying over the local tensor slice, which is at start_ring_index. If clockwise, then dir1 will be anticlockwise, which means that the ring index will update in ascending order.
    // Therefore, to undo that, we subtract 1. If anticlockwise, then dir1 will be clockwise, which means that the ring index will update in descending order. Therefore, to undo that, we add 1.
    uint32_t ring_index_dir1 = (is_clockwise_direction ? start_ring_index - 1 : start_ring_index + 1) % ring_size;

    volatile tt_l1_ptr uint32_t* signal_op_semaphore_ptrs[2] = {signal_op_semaphore_addr_ptr_dir0, signal_op_semaphore_addr_ptr_dir1};
    uint32_t ring_idxs[2] = {ring_index_dir0, ring_index_dir1};
    uint32_t start_page_idxs[2] = {ring_index_dir0 * output_page_offset, ring_index_dir1 * output_page_offset};
    uint32_t is_clockwise_dirs[2] = {is_clockwise_direction, !is_clockwise_direction};

    const uint32_t num_blocks_per_slice = tensor_slice_shape_width / in0_block_w; // TODO: Confirm if tensor_slice_shape_width is in tiles
    DPRINT << "CHECK num_blocks_per_slice * num tensor slices ::: " << num_blocks_per_slice * (num_transfers * 2) << " === " << num_blocks << ENDL();

    // // DPRINT the num blocks stuff
    // DPRINT << "Num blocks: " << num_blocks << ENDL();
    // DPRINT << "Num blocks per slice: " << num_blocks_per_slice << ENDL();
    // DPRINT << "in0_block_w: " << in0_block_w << ENDL();
    // DPRINT << "Num transfers: " << num_transfers << ENDL();

#endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;

#ifdef IN0_SHARDED
    // In case we need to send multiple blocks per shard, in0 sharded cb is cb2 and we extract the sub-blocks to cb0
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    uint64_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        constexpr uint32_t cb_id_in2 = 2;  // in0 sharded cb if extract_shard_sub_blocks
        noc_shard_read_start_addr = get_noc_addr(get_read_ptr(cb_id_in2));
    } else {
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
#else
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = in0_single_tile_size_bytes, .data_format = in0_data_format};
#endif

#ifndef SKIP_MCAST
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

#ifdef IN0_SHARDED
    uint32_t in0_start_address = get_write_ptr(cb_id_in0);
#endif
#endif

    for (uint32_t b = 0; b < batch; ++b) {
#ifndef MATMUL_SIGNAL
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
#else
        // (num_transfers * 2) * num_blocks_per_slice === num_blocks so we iterate through the same number of blocks
        for (uint32_t i = 0, dir = 0; i < num_transfers * 2; i++, dir = !dir) {
            uint32_t tensor_slice_cnt = i / 2; // Since we are alternating between the two directions, we need to divide by 2 to get the correct tensor slice count in each direction

            // Note: this call will only run when block is aligned to the start of a tensor slice
            advance_start_page_idx(start_page_idxs[dir], ring_idxs[dir], ring_size, is_clockwise_dirs[dir], output_page_offset, last_output_page_offset);

            uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id + start_page_idxs[dir];

            // Wait for the signal for this tensor slice
            // DPRINT << "Waiting for signal for tensor slice: " << tensor_slice_cnt << " in direction: " << dir << ENDL();
            if ((!dir && tensor_slice_cnt < num_transfers) || (dir && tensor_slice_cnt < num_transfers - 1)) { // Using dir as a selector to select which logic to choose, because dir = 1 will have 1 less semaphore (because one is local already)
                noc_semaphore_wait_min(signal_op_semaphore_ptrs[dir], tensor_slice_cnt + 1); // TODO: Update the semaphore pointer to be an array based on direction, just like datacopy
            }

            for (uint32_t tensor_slice_block = 0; tensor_slice_block < num_blocks_per_slice; ++tensor_slice_block) {
#endif
#ifndef IN0_SHARDED
            // Operand 0
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

#ifndef SKIP_MCAST
            uint32_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif

            // Copy in0 block into CB, as the default kernel
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
                    if (h < last_block_h) {
                        noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    }
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();
#else
            if constexpr (extract_shard_sub_blocks) {
                // Operand 0
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

#ifndef SKIP_MCAST
                in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif

                uint64_t noc_shard_read_addr = noc_shard_read_start_addr;
                noc_shard_read_start_addr += shard_read_width;

                for (uint32_t i = 0; i < shard_height_in_tiles; i++) {
                    noc_async_read(noc_shard_read_addr, l1_write_addr_in0, shard_read_width);

                    l1_write_addr_in0 += shard_read_width;
                    noc_shard_read_addr += shard_read_stride;
                }

                noc_async_read_barrier();
            }
#endif

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
                in0_mcast_receiver_semaphore_addr,
                in0_mcast_receiver_semaphore_noc_addr,
                in0_mcast_num_cores);
#endif

#ifndef IN0_SHARDED
            cb_push_back(cb_id_in0, in0_block_num_tiles);
#else
            if constexpr (extract_shard_sub_blocks) {
                cb_push_back(cb_id_in0, in0_block_num_tiles);
            }
#endif
#ifdef MATMUL_SIGNAL
            } // end of for loop that iterates through num blocks per slice
        } // end of for loop that iterates through num transfers in both directions
#else
        } // end of for loop that iterates through num blocks
#endif
        in0_tensor_start_tile_id += MtKt;
    }
}
