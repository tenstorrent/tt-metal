// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    // READER
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(2);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(3);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(4);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(5);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(6);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(7);

    // padding args (READER)
    const uint32_t last_block_w = get_arg_val<uint32_t>(8);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(9);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(10);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(11);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(12);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(13);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(14);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(15);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool in1_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;

    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(4);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(7);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // in1 mcast args
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(11);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(12);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(13);
    constexpr uint32_t batch = get_compile_time_arg_val(14);
    constexpr uint32_t bcast_B = get_compile_time_arg_val(15);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(17);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(18);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(19);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(20);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(21);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(22);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(23);  // if 0
    // Don't need batch; same as batch from READER args

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(16);
    const uint32_t in3_tensor_start_tile_id = get_arg_val<uint32_t>(17);

    constexpr bool in3_is_dram = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t in3_tensor_stride_w = get_compile_time_arg_val(25);

    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(cb_id_in3);
    constexpr DataFormat bias_data_format = get_dataformat(cb_id_in3);

    uint32_t l1_write_addr_in3;

    const InterleavedAddrGenFast<in3_is_dram> s3 = {
        .bank_base_address = in3_tensor_addr,
        .page_size = bias_single_tile_size_bytes,
        .data_format = bias_data_format};
#endif

// RT and COMPILE TIME ARGS for DRAM sharded weights
#ifdef IN1_DRAM_SHARDED
    const uint32_t vc = get_arg_val<uint32_t>(18);
    const uint32_t num_dram_shards_to_read = get_arg_val<uint32_t>(19);
    const uint32_t dram_tensor_start_offset = get_arg_val<uint32_t>(20);
    tt_l1_ptr uint32_t* in1_block_w_dram_stride_bytes = (tt_l1_ptr uint32_t*)get_arg_addr(21);
    tt_l1_ptr uint32_t* current_dram_bank_id = (tt_l1_ptr uint32_t*)get_arg_addr(22);

    constexpr uint32_t in1_dram_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t in1_block_w_dram_bytes = get_compile_time_arg_val(27);
#endif

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

// COMPILE TIME ARGS for All Gather Matmul
#ifdef MATMUL_SIGNAL
    /* Overlapped all-gather with matmul params */
    const uint32_t num_transfers = get_compile_time_arg_val(26);
    const uint32_t ring_size = get_compile_time_arg_val(27);
    const uint32_t start_ring_index = get_compile_time_arg_val(28);
    const uint32_t tensor_slice_shape_width = get_compile_time_arg_val(29);
    const uint32_t output_page_offset = get_compile_time_arg_val(30);
    const uint32_t last_output_page_offset = get_compile_time_arg_val(31);
    const uint32_t is_clockwise_direction = get_compile_time_arg_val(32);
    const uint32_t signal_op_sem_addr_dir0 = get_compile_time_arg_val(33);
    const uint32_t signal_op_sem_addr_dir1 = get_compile_time_arg_val(34);

    // // DPRINT all the above values
    // DPRINT << "num_transfers: " << num_transfers << ENDL();
    // DPRINT << "ring_size: " << ring_size << ENDL();
    // DPRINT << "start_ring_index: " << start_ring_index << ENDL();
    // DPRINT << "tensor_slice_shape_width: " << tensor_slice_shape_width << ENDL();
    // DPRINT << "output_page_offset: " << output_page_offset << ENDL();
    // DPRINT << "last_output_page_offset: " << last_output_page_offset << ENDL();
    // DPRINT << "is_clockwise_direction: " << is_clockwise_direction << ENDL();
    // DPRINT << "signal_op_sem_addr_dir0: " << signal_op_sem_addr_dir0 << ENDL();
    // DPRINT << "signal_op_sem_addr_dir1: " << signal_op_sem_addr_dir1 << ENDL();


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

    const uint32_t num_blocks_per_slice = tensor_slice_shape_width / in1_block_h; // TODO: Confirm if tensor_slice_shape_width is in tiles
    DPRINT << "IN1 CHECK num_blocks_per_slice * num tensor slices ::: " << num_blocks_per_slice * (num_transfers * 2) << " === " << num_blocks << ENDL();

    // // DPRINT the num blocks stuff
    // DPRINT << "Num blocks: " << num_blocks << ENDL();
    // DPRINT << "Num blocks per slice: " << num_blocks_per_slice << ENDL();
    // DPRINT << "tensor_slice_shape_width: " << tensor_slice_shape_width << ENDL();
    // DPRINT << "in1_block_w: " << in1_block_w << ENDL();
    // DPRINT << "Num transfers: " << num_transfers << ENDL();
#endif


//  READER
#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, in1_block_num_tiles);
    cb_push_back(cb_id_in1, in1_block_num_tiles);
#else
    uint32_t l1_write_addr_in1;

    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = in1_single_tile_size_bytes, .data_format = in1_data_format};
#endif

    //  WRITER
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr DataFormat output_data_format = get_dataformat(cb_id_out0);
    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr,
        .page_size = output_single_tile_size_bytes,
        .data_format = output_data_format};

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in1_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_sender_semaphore_addr);

    const uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_receiver_semaphore_addr);

    const uint64_t in1_multicast_data_noc = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, in1_mcast_dest_noc_start_y, in1_mcast_dest_noc_end_x, in1_mcast_dest_noc_end_y, 0);
#ifdef IN1_SHARDED
    uint64_t in1_start_address = get_write_ptr(cb_id_in1);
#endif
#endif

#ifdef IN1_DRAM_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif

    for (uint32_t b = 0; b < batch; ++b) {
#ifdef IN1_DRAM_SHARDED
        uint32_t l1_read_addr_in1_offset = 0;
#endif
#ifndef MATMUL_SIGNAL
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
#else
        // (num_transfers * 2) * num_blocks_per_slice === num_blocks so we iterate through the same number of blocks
        for (uint32_t i = 0, dir = 0; i < num_transfers * 2; i++, dir = !dir) {
            uint32_t tensor_slice_cnt = i / 2; // Since we are alternating between the two directions, we need to divide by 2 to get the correct tensor slice count in each direction

            // Note: this call will only run when block is aligned to the start of a tensor slice
            advance_start_page_idx(start_page_idxs[dir], ring_idxs[dir], ring_size, is_clockwise_dirs[dir], output_page_offset, last_output_page_offset);

            uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id + start_page_idxs[dir];

            // Wait for the signal for this tensor slice
            // DPRINT << "Waiting for signal for tensor slice: " << tensor_slice_cnt << " in direction: " << dir << ENDL();
            if ((!dir && tensor_slice_cnt < num_transfers) || (dir && tensor_slice_cnt < num_transfers - 1)) { // Using dir as a selector to select which logic to choose, because dir = 1 will have 1 less semaphore (because one is local already)
                // noc_semaphore_wait_min(signal_op_semaphore_ptrs[dir], tensor_slice_cnt + 1); // TODO: Update the semaphore pointer to be an array based on direction, just like datacopy
            }

            for (uint32_t tensor_slice_block = 0; tensor_slice_block < num_blocks_per_slice; ++tensor_slice_block) {
#endif
#ifdef IN1_DRAM_SHARDED
            // Operand 1
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            uint64_t in1_start_address =
                get_write_ptr(cb_id_in1);  // copy start address of block, to be used for mcasting

            uint32_t l1_write_addr_in1_offset = 0;
            uint32_t next_bank_id_and_dram_stride_index = 0;

            for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                uint32_t in1_base_addr = noc_async_read_tile_dram_sharded_set_state<in1_single_tile_size_bytes, true>(
                    in1_tensor_addr, current_dram_bank_id[next_bank_id_and_dram_stride_index], vc);

                if (i == 0) {
                    in1_base_addr += dram_tensor_start_offset;
                }

                uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1) + l1_write_addr_in1_offset;
                uint32_t in1_block_w_dram =
                    in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] / in1_single_tile_size_bytes;

                for (uint32_t m = 0; m < in1_block_h; ++m) {
                    uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                    uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
                    for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                        noc_async_read_tile_dram_sharded_with_state(
                            in1_base_addr, l1_read_addr_in1_temp, l1_write_addr_in1_temp);
                        l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                        l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                    }
                    l1_read_addr_in1 += in1_block_w_dram_bytes;
                    l1_write_addr_in1 += in1_block_w_bytes;
                }
                l1_write_addr_in1_offset += in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                next_bank_id_and_dram_stride_index += 2;
            }
            l1_read_addr_in1_offset += in1_dram_block_size_bytes;
            noc_async_read_barrier();
#else
#ifndef IN1_SHARDED
            // Operand 1
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            uint64_t in1_start_address = l1_write_addr_in1;  // copy start address of block, to be used for mcasting

            // Copy in1 block into CB, as the default kernel
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
                    if (w < last_block_w) {
                        noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    }
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();
#endif
#endif  // IN1_DRAM_SHARDED

#ifndef SKIP_MCAST
            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value
            // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
            noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
            noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in1_multicast_data_addr = in1_multicast_data_noc | in1_start_address;

            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_cores, true, true);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same
            // cmd_buf Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                in1_mcast_receiver_semaphore_addr,
                in1_mcast_receiver_semaphore_noc_addr,
                in1_mcast_num_cores);
#endif

#ifndef IN1_SHARDED
            cb_push_back(cb_id_in1, in1_block_num_tiles);
#endif

#ifdef MATMUL_SIGNAL
            }
        }
#else
        }
#endif
#ifdef FUSE_BIAS
        // Only read bias on first batch
        if (b == 0) {
            // Operand 1
            cb_reserve_back(cb_id_in3, in1_block_w);
            l1_write_addr_in3 = get_write_ptr(cb_id_in3);

            uint64_t in3_start_address = l1_write_addr_in3;  // copy start address of block, to be used for mcasting
            uint32_t in3_block_size_bytes = 0;               // can be optimized later, pass it to kernel

#ifdef IN1_DRAM_SHARDED
            uint32_t l1_write_addr_in3_offset = 0;
            uint32_t next_bank_id_and_dram_stride_index = 0;

            for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                uint32_t in3_base_addr = noc_async_read_tile_dram_sharded_set_state<bias_single_tile_size_bytes, true>(
                    in3_tensor_addr, current_dram_bank_id[next_bank_id_and_dram_stride_index], vc);

                if (i == 0) {
                    in3_base_addr += dram_tensor_start_offset;
                }

                uint32_t l1_read_addr_in3 = 0;
                uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3) + l1_write_addr_in3_offset;
                uint32_t in3_block_w_dram =
                    in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] / bias_single_tile_size_bytes;

                for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                    noc_async_read_tile_dram_sharded_with_state(in3_base_addr, l1_read_addr_in3, l1_write_addr_in3);
                    l1_read_addr_in3 += bias_single_tile_size_bytes;
                    l1_write_addr_in3 += bias_single_tile_size_bytes;
                    in3_block_size_bytes += bias_single_tile_size_bytes;
                }
                l1_write_addr_in3_offset += in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                next_bank_id_and_dram_stride_index += 2;
            }
            noc_async_read_barrier();
#else
            // Copy in1 block into CB, as the default kernel
            uint32_t in3_tensor_tile_id = in3_tensor_start_tile_id;
            for (uint32_t w = 0; w < in1_block_w; ++w) {
                if (w < last_block_w) {
                    noc_async_read_tile(in3_tensor_tile_id, s3, l1_write_addr_in3);
                }
                l1_write_addr_in3 += bias_single_tile_size_bytes;
                in3_tensor_tile_id += in3_tensor_stride_w;
                in3_block_size_bytes += bias_single_tile_size_bytes;
            }
            // Barrier! make sure the reads are done
            noc_async_read_barrier();
#endif

#ifndef SKIP_MCAST

            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value
            // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
            noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
            noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in3_multicast_data_addr = in1_multicast_data_noc | in3_start_address;

            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                in3_start_address, in3_multicast_data_addr, in3_block_size_bytes, in1_mcast_num_cores, true, true);
            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same
            // cmd_buf Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                in1_mcast_receiver_semaphore_addr,
                in1_mcast_receiver_semaphore_noc_addr,
                in1_mcast_num_cores);
#endif

            cb_push_back(cb_id_in3, in1_block_w);
        }
#endif
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }

#ifndef OUT_SHARDED
        // WRITER
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h; ++sbh) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w; ++sbw) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                uint32_t out_subblock_h_ = out_subblock_h;
                uint32_t out_subblock_w_ = out_subblock_w;
                uint32_t subblock_tiles_addr_skip = 0;
                if (sbh == out_num_nonzero_subblocks_h - 1) {
                    out_subblock_h_ = out_last_subblock_h;
                }
                if (sbw == out_num_nonzero_subblocks_w - 1) {
                    out_subblock_w_ = out_last_subblock_w;
                    subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                }

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);

                        l1_read_addr += output_single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    // Skip padded tiles in subblock along row
                    l1_read_addr += subblock_tiles_addr_skip;
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            // Pop fully padded subblocks along the row
            cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
            cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // Pop row(s) of fully padded subblocks
        cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
        cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
        out_tensor_start_tile_id += MtNt;
#endif
    }

#if OUT_SHARDED
    cb_wait_front(
        cb_id_out0,
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
}
