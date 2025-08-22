// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
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

    constexpr bool is_clockwise_direction = get_compile_time_arg_val(12);  // Specify direction for the first half
    const uint32_t signal_op_sem_addr_dir0 = get_semaphore(get_compile_time_arg_val(13));
    const uint32_t signal_op_sem_addr_dir1 = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t max_buffer_size = get_compile_time_arg_val(15);

    // Compile time args for matmul signal semaphore
    constexpr uint32_t num_matmul_cores_to_signal = get_compile_time_arg_val(16);

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* matmul_signal_sems =
        (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, 2));  // Matmul signal semaphore address
    const uint32_t* matmul_cores_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(
        rt_args_idx, 2 * num_matmul_cores_to_signal));  // Matmul core NOC coordinates [x1, y1, x2, y2...]

    // Setup buffers
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    // DRAM reader in
    InterleavedAddrGenFast<in_is_dram> d_in = {
        .bank_base_address = dram_buffer_src_addr, .page_size = page_size, .data_format = in0_df};

    // DRAM writer out
    InterleavedAddrGenFast<out_is_dram> d_out = {
        .bank_base_address = dram_buffer_dst_addr, .page_size = page_size, .data_format = in0_df};

    // Internal semaphores
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir0 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir0);
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr_dir1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_op_sem_addr_dir1);

    // Args used to read/write slices of the tensor
    const uint32_t num_pages = tensor_slice_shape_width * tensor_slice_shape_height;

    ttnn::ccl::coord_t tensor_shape = {tensor_shape_width, tensor_shape_height};
    ttnn::ccl::coord_t tensor_slice_shape = {tensor_slice_shape_width, tensor_slice_shape_height};

    uint32_t ring_index_dir0 = start_ring_index;
    uint32_t ring_index_dir1 = start_ring_index;

    uint32_t start_page_idx_dir0 = ring_index_dir0 * output_page_offset;
    uint32_t start_page_idx_dir1 = ring_index_dir1 * output_page_offset;

    volatile tt_l1_ptr uint32_t* signal_op_semaphore_ptrs[2] = {
        signal_op_semaphore_addr_ptr_dir0, signal_op_semaphore_addr_ptr_dir1};
    uint32_t start_page_idxs[2] = {start_page_idx_dir0, start_page_idx_dir1};
    uint32_t ring_idxs[2] = {ring_index_dir0, ring_index_dir1};
    uint32_t is_clockwise_dirs[2] = {is_clockwise_direction, !is_clockwise_direction};

    // Main for loop where each iteration handles a tensor slice
    // The loop alternates between the two directions, hence it runs for double the number of transfers
    for (uint32_t i = 0, dir = 1; i < num_transfers * 2; i++, dir = !dir) {
        uint32_t tensor_slice_cnt = i / 2;  // Since we are alternating between the two directions, we need to divide by
                                            // 2 to get the correct tensor slice count in each direction

        // Update location in input and output tensor in DRAM
        if (i > 0) {  // Skip update for local tensor slice
            advance_start_page_idx(
                start_page_idxs[dir],
                ring_idxs[dir],
                ring_size,
                is_clockwise_dirs[dir],
                output_page_offset,
                last_output_page_offset);
        }

        uint32_t curr_page_in_idx = start_page_idxs[dir];
        uint32_t curr_page_out_idx = start_page_idxs[dir];

        uint32_t offset_into_in_tensor_slice = 0;
        uint32_t offset_into_out_tensor_slice = 0;

        bool last_page_of_in_tensor_slice = false;
        bool last_page_of_out_tensor_slice = false;

        ttnn::ccl::coord_t offset_worker_slice = {0, 0};

        // DPRINT << "WAITING FOR OP SIGNAL IN DATACOPY" << ENDL();
        noc_semaphore_wait_min(signal_op_semaphore_ptrs[dir], tensor_slice_cnt + 1);
        // DPRINT << "RECEIVED OP SIGNAL IN DATACOPY" << ENDL();

        // Signal matmul to begin
        for (uint32_t i = 0; i < num_matmul_cores_to_signal; i++) {
            auto& matmul_core_noc_x = matmul_cores_noc_coords[i * 2];
            auto& matmul_core_noc_y = matmul_cores_noc_coords[i * 2 + 1];
            auto remote_matmul_signal_sem_addr =
                get_noc_addr(matmul_core_noc_x, matmul_core_noc_y, get_semaphore(matmul_signal_sems[dir]));
            noc_semaphore_inc(remote_matmul_signal_sem_addr, 1);
        }

        // To account for the granularity based on restrictions on the buffer size of L1
        for (uint32_t pages = 0; pages < num_pages;) {
            uint32_t num_pages_to_transfer = std::min(num_pages - pages, max_buffer_size);
            // DPRINT << "num_pages_to_transfer: " << num_pages_to_transfer << ENDL();

            // Read the data from DRAM into cb0
            read_wrapped_chunk_from_output_tensor(  // Use this function assuming that the worker slice is the same as
                                                    // the tensor slice
                curr_page_in_idx,
                offset_into_in_tensor_slice,  // Viewing this as offset into the entire tensor slice since there's only
                                              // one worker
                offset_worker_slice,          // offset_worker_slice is always 0 since there's only one worker
                tensor_slice_shape,  // worker_slice_shape should be same as tensor_slice_shape since there's only one
                                     // worker

                tensor_shape,
                tensor_slice_shape,
                cb_id_in0,
                d_in,
                num_pages_to_transfer,
                page_size,
                last_page_of_in_tensor_slice);

            // Write the data from cb0 to DRAM
            write_wrapped_chunk(
                curr_page_out_idx,
                offset_into_out_tensor_slice,  // Viewing this as offset into the entire tensor slice since there's only
                                               // one worker
                offset_worker_slice,           // offset_worker_slice is always 0 since there's only one worker
                tensor_slice_shape,  // worker_slice_shape should be same as tensor_slice_shape since there's only one
                                     // worker

                tensor_shape,
                tensor_slice_shape,
                cb_id_in0,
                d_out,
                num_pages_to_transfer,
                page_size,
                last_page_of_out_tensor_slice);

            pages += num_pages_to_transfer;

            // Push and pop filler pages if needed to align CB ptr
            if (num_pages_to_transfer < max_buffer_size) {
                push_filler_pages_to_cb(cb_id_in0, max_buffer_size - num_pages_to_transfer);
                pop_filler_pages_from_cb(cb_id_in0, max_buffer_size - num_pages_to_transfer);
            }
        }
    }
}
