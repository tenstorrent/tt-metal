// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "api/compile_time_args.h"
#include <api/dataflow/dataflow_api.h>
#include "ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"
#include "../grid_sample_reader_common.hpp"

#define PRINT_AND_PROFILE 0
#if PRINT_AND_PROFILE
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#endif

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t num_pages = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);

    constexpr uint32_t input_height = get_compile_time_arg_val(6);
    constexpr uint32_t input_width = get_compile_time_arg_val(7);
    constexpr uint32_t grid_batches = get_compile_time_arg_val(8);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(9);
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(10);
    constexpr bool use_precomputed_grid = get_compile_time_arg_val(11);

    constexpr auto src_args = TensorAccessorArgs<12>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto grid_tensor_accessor = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    experimental::CB grid_cb(grid_cb_index);
    experimental::Noc noc;

    const uint32_t end_id = start_page_id + num_pages;

    experimental::CB input_cb(input_cb_index);
    experimental::CB scalar_cb(scalar_cb_index);
    zero_out_tiles<input_cb_index>(noc, input_cb);

    // Calculate starting batch from starting spatial position (avoid division in loop)
    uint32_t curr_batch = start_page_id / output_hw_size;
    uint32_t spatial_points_processed = start_page_id % output_hw_size;
    uint32_t batch_offset = curr_batch * input_height * input_width;

    // Outer loop: iterate over spatial positions (output sticks)
    for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
        // Read the grid stick for this spatial position (contains grid_batches sets of coordinates)
        noc.async_read(grid_tensor_accessor, grid_cb, grid_stick_nbytes, {.page_id = spatial_pos}, {});
        noc.async_read_barrier();

        // Cast to appropriate pointer type for grid data access
        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(grid_cb.get_write_ptr());

        // Inner loop: process grid_batches coordinate sets within this spatial position
        for (uint32_t grid_idx = 0; grid_idx < grid_batches; ++grid_idx) {
            // Direct template dispatch - no branching needed
            process_grid_point<
                grid_dtype,
                use_precomputed_grid,
                input_height,
                input_width,
                input_stick_nbytes,
                input_cb_index,
                scalar_cb_index>(noc, input_cb, scalar_cb, grid_ptr, grid_idx, input_tensor_accessor, batch_offset);
        }

        // Update batch tracking (avoid division in loop)
        ++spatial_points_processed;
        if (spatial_points_processed == output_hw_size) {
            spatial_points_processed = 0;
            ++curr_batch;
            batch_offset = curr_batch * input_height * input_width;
        }
    }
}
