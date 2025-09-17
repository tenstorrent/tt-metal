// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "compile_time_args.h"
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/height_sharded_reader_common.hpp"
#include "../grid_sample_reader_common.hpp"

#define PRINT_AND_PROFILE 0
#if PRINT_AND_PROFILE
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"
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
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t grid_batches = get_compile_time_arg_val(7);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(9);
    constexpr bool use_precomputed_grid = get_compile_time_arg_val(10);

    constexpr auto src_args = TensorAccessorArgs<11>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto grid_tensor_accessor = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    const uint32_t end_id = start_page_id + num_pages;

    /*
    In the case of grid sampling, we need to account for the fact that the grid coordinates may fall outside the bounds
    of the input image. Since the padding mode is zero, we would simply set the weights for the appropriate sticks to
    zero in the for loop, and simply do not read from DRAM. In that case the stick we send to reduction would be the
    last pixel that we read for the appropriate location (SW, SE, NW, NE), but since weights are 0 this is not a
    problem.

    However, if there was no previous read for the appropriate stick, the memory in that location is invalid, and could
    include NaN and Inf values. For that reason we zero out the input_cb at the start.
    */

    zero_out_tiles<input_cb_index>();

    // Calculate starting batch from starting spatial position (avoid division in loop)
    uint32_t curr_batch = start_page_id / output_hw_size;
    uint32_t spatial_points_processed = start_page_id % output_hw_size;
    uint32_t batch_offset = curr_batch * input_height * input_width;

    // Outer loop: iterate over spatial positions (output sticks)
    for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
        // Read the grid stick for this spatial position (contains grid_batches sets of coordinates)
        uint32_t l1_write_grid_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = grid_tensor_accessor.get_noc_addr(spatial_pos);

        noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_nbytes);
        noc_async_read_barrier();

        // Cast to appropriate pointer type for grid data access
        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_grid_addr);

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
                scalar_cb_index>(grid_ptr, grid_idx, input_tensor_accessor, batch_offset);
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
