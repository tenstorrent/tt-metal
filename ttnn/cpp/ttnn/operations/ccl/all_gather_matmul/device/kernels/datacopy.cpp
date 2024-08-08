// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "debug/dprint.h"

void kernel_main() {

    // Compile time Args
    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t num_transfers = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);

    constexpr uint32_t start_ring_index = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);

    // Get the size of the global output tensor
    constexpr uint32_t tensor_shape_width = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_shape_height = get_compile_time_arg_val(7);
    // Get the size of a tensor slice in a global output tensor
    constexpr uint32_t tensor_slice_shape_width = get_compile_time_arg_val(8);
    constexpr uint32_t tensor_slice_shape_height = get_compile_time_arg_val(9);

    // Get the offsets necessary to stride through global output tensor
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(10);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(11);

    constexpr bool is_clockwise_direction = get_compile_time_arg_val(12) == 1; // Specify direction for the first half
    constexpr uint32_t signal_op_sem_addr_dir0 = get_compile_time_arg_val(13);
    constexpr uint32_t signal_op_sem_addr_dir1 = get_compile_time_arg_val(14);


    // Runtime args
    const uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(1);


    // Setup buffers
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    // DRAM reader in
    InterleavedAddrGenFast<in_is_dram> d_in= {
        .bank_base_address = dram_buffer_src_addr,
        .page_size = page_size,
        .data_format = in0_df
    };


    // DRAM writer out
    InterleavedAddrGenFast<out_is_dram> d_out = {
        .bank_base_address = dram_buffer_dst_addr,
        .page_size = page_size,
        .data_format = in0_df
    };

    // Internal semaphores
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir0);
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir1);


    // Args used to read/write slices of the tensor
    const uint32_t num_pages = tensor_slice_shape_width * tensor_slice_shape_height;
    const uint32_t max_buffer_size = 200; // TODO: Update to be the actual buffer size of L1

    ttnn::ccl::coord_t tensor_shape = {tensor_shape_width, tensor_shape_height};
    ttnn::ccl::coord_t tensor_slice_shape = {tensor_slice_shape_width, tensor_slice_shape_height};

    uint32_t ring_index_dir0 = start_ring_index;
    // Adjust to include copying over the local tensor slice, which is at start_ring_index. If clockwise, then dir1 will be anticlockwise, which means that the ring index will update in ascending order.
    // Therefore, to undo that, we subtract 1. If anticlockwise, then dir1 will be clockwise, which means that the ring index will update in descending order. Therefore, to undo that, we add 1.
    uint32_t ring_index_dir1 = (is_clockwise_direction ? start_ring_index - 1 : start_ring_index + 1) % ring_size;

    uint32_t start_page_idx_dir0 = ring_index_dir0 * output_page_offset;
    uint32_t start_page_idx_dir1 = ring_index_dir1 * output_page_offset;

    uint32_t cb_write_addr = get_write_ptr(cb_id_in0);
    uint32_t cb_read_addr = get_read_ptr(cb_id_in0);


    volatile tt_l1_ptr uint32_t* signal_op_semaphore_ptrs[2] = {signal_op_semaphore_addr_ptr_dir0, signal_op_semaphore_addr_ptr_dir1};
    uint32_t start_page_idxs[2] = {start_page_idx_dir0, start_page_idx_dir1};
    uint32_t ring_idxs[2] = {ring_index_dir0, ring_index_dir1};
    uint32_t is_clockwise_dirs[2] = {is_clockwise_direction, !is_clockwise_direction};

    // Main for loop where each iteration handles a tensor slice
    // The loop alternates between the two directions, hence it runs for double the number of transfers
    for (uint32_t i = 0, dir = 0; i < num_transfers * 2; i++, dir = !dir) {
        uint32_t tensor_slice_cnt = i / 2; // Since we are alternating between the two directions, we need to divide by 2 to get the correct tensor slice count in each direction

        // Update location in input and output tensor in DRAM
        advance_start_page_idx(start_page_idxs[dir], ring_idxs[dir], ring_size, is_clockwise_dirs[dir], output_page_offset, last_output_page_offset);

        // DPRINT << "DIRECTION 0 RING INDEX>>>> " << ring_index_dir0 << ENDL();

        uint32_t curr_page_in_idx = start_page_idxs[dir];
        uint32_t curr_page_out_idx = start_page_idxs[dir];

        uint32_t offset_into_in_tensor_slice = 0;
        uint32_t offset_into_out_tensor_slice = 0;

        bool last_page_of_in_tensor_slice = false;
        bool last_page_of_out_tensor_slice = false;

        ttnn::ccl::coord_t offset_worker_slice = {0, 0};

        // DPRINT << "WAITING FOR OP SIGNAL IN DATACOPY" << ENDL();
        if ((!dir && tensor_slice_cnt < num_transfers) || (dir && tensor_slice_cnt < num_transfers - 1)) { // Using dir as a selector to select which logic to choose, because dir = 1 will have 1 less semaphore (because one is local already)
            noc_semaphore_wait_min(signal_op_semaphore_ptrs[dir], tensor_slice_cnt + 1);
        }
        // DPRINT << "RECEIVED OP SIGNAL IN DATACOPY" << ENDL();

        // To account for the granularity based on restrictions on the buffer size of L1
        for (uint32_t pages = 0; pages < num_pages;) {
            uint32_t num_pages_to_transfer = std::min(num_pages - pages, max_buffer_size);
            // DPRINT << "num_pages_to_transfer: " << num_pages_to_transfer << ENDL();

            // Read the data from DRAM into cb0
            read_wrapped_chunk_from_output_tensor( // Use this function assuming that the worker slice is the same as the tensor slice
                curr_page_in_idx,
                offset_into_in_tensor_slice, // Viewing this as offset into the entire tensor slice since there's only one worker
                offset_worker_slice, // offset_worker_slice is always 0 since there's only one worker
                tensor_slice_shape, // worker_slice_shape should be same as tensor_slice_shape since there's only one worker

                tensor_shape,
                tensor_slice_shape,
                cb_id_in0,
                d_in,
                num_pages_to_transfer,
                page_size,
                last_page_of_in_tensor_slice,
                cb_write_addr);


            // Write the data from cb0 to DRAM
            write_wrapped_chunk(
                curr_page_out_idx,
                offset_into_out_tensor_slice, // Viewing this as offset into the entire tensor slice since there's only one worker
                offset_worker_slice, // offset_worker_slice is always 0 since there's only one worker
                tensor_slice_shape, // worker_slice_shape should be same as tensor_slice_shape since there's only one worker

                tensor_shape,
                tensor_slice_shape,
                cb_id_in0,
                d_out,
                num_pages_to_transfer,
                page_size,
                last_page_of_out_tensor_slice,
                cb_read_addr);

            pages += num_pages_to_transfer;
        }

    }


}
