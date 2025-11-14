// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified kernel for nearest neighbor grid sampling - INTERLEAVED MODE ONLY
// This kernel does the complete operation: read input → sample nearest → write output to DRAM
// Both "reader" and "writer" cores run this same kernel on different data chunks (split reader)

#include <cmath>
#include <stdint.h>
#include "compile_time_args.h"
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
#include "../grid_sample_reader_common.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t grid_addr = get_arg_val<uint32_t>(2);
    uint32_t grid_sticks = get_arg_val<uint32_t>(3);
    uint32_t grid_start_id = get_arg_val<uint32_t>(4);    // Starting grid stick index for this core
    uint32_t output_start_id = get_arg_val<uint32_t>(5);  // Starting output stick index for this core

    // Compile-time arguments
    constexpr uint32_t work_cb_index = get_compile_time_arg_val(0);  // Temporary work buffer in L1
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);  // Temporary grid buffer in L1
    constexpr uint32_t input_stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t grid_stick_size = get_compile_time_arg_val(3);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t grid_batching_factor = get_compile_time_arg_val(7);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t grid_hw = get_compile_time_arg_val(9);
    constexpr bool use_precomputed_grid = get_compile_time_arg_val(10);
    constexpr bool enable_split_reader = get_compile_time_arg_val(11);
    constexpr uint32_t reader_id = get_compile_time_arg_val(12);

    // TensorAccessor arguments
    constexpr auto input_args = TensorAccessorArgs<13>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto grid_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const auto input_tensor_accessor = TensorAccessor(input_args, input_addr, input_stick_size);
    const auto output_tensor_accessor = TensorAccessor(output_args, output_addr, output_stick_size);
    const auto grid_tensor_accessor = TensorAccessor(grid_args, grid_addr, grid_stick_size);

    // Grid coordinate scaling factors (align_corners=False by default)
    constexpr float input_height_f = static_cast<float>(input_height);
    constexpr float input_width_f = static_cast<float>(input_width);
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    const uint32_t grid_end_id = grid_start_id + grid_sticks;

    // Calculate starting position and step based on split reader configuration
    // When split reader enabled:
    //   - Reader 0 (RISCV_0) processes even-indexed grid points: 0, 2, 4, ...
    //   - Reader 1 (RISCV_1) processes odd-indexed grid points: 1, 3, 5, ...
    uint32_t spatial_pos = grid_start_id;
    uint32_t grid_advance_step = 1;
    uint32_t output_stick_id = output_start_id;

    if constexpr (enable_split_reader) {
        // Reader 1 starts at the next grid point (odd indices)
        if constexpr (reader_id == 1) {
            spatial_pos += 1;
            output_stick_id += grid_batching_factor;  // Output also offset by one batch
        }
        grid_advance_step = 2;  // Both readers advance by 2 to process alternating points
    }

    // Main loop: process grid sticks assigned to this core
    for (; spatial_pos < grid_end_id; spatial_pos += grid_advance_step) {
        // Calculate current batch and offset for this spatial position
        const uint32_t curr_batch = spatial_pos / grid_hw;
        const uint32_t batch_offset = curr_batch * input_height * input_width;
        // Read grid stick from DRAM to L1
        uint32_t l1_write_grid_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = grid_tensor_accessor.get_noc_addr(spatial_pos);
        noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_size);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_grid_addr);

        // Use work CB as scratch space (no need for reserve/pop since same kernel writes and reads)
        uint32_t l1_write_work_addr = get_write_ptr(work_cb_index);

        // Process each grid point in this stick
        for (uint32_t grid_idx = 0; grid_idx < grid_batching_factor; ++grid_idx) {
            int32_t h_nearest, w_nearest;

            // Compile-time dispatch based on grid configuration
            // Use simplified nearest neighbor coordinate reader (no weights needed)
            if constexpr (use_precomputed_grid) {
                GridCoordinateReaderNearest<DTYPE_BFLOAT16, true>::read_grid_point_nearest(
                    grid_ptr,
                    grid_idx,
                    height_scale,
                    height_offset,
                    width_scale,
                    width_offset,
                    input_height,
                    input_width,
                    h_nearest,
                    w_nearest);
            } else if constexpr (grid_dtype == DTYPE_FLOAT32) {
                GridCoordinateReaderNearest<DTYPE_FLOAT32, false>::read_grid_point_nearest(
                    grid_ptr,
                    grid_idx,
                    height_scale,
                    height_offset,
                    width_scale,
                    width_offset,
                    input_height,
                    input_width,
                    h_nearest,
                    w_nearest);
            } else {
                GridCoordinateReaderNearest<DTYPE_BFLOAT16, false>::read_grid_point_nearest(
                    grid_ptr,
                    grid_idx,
                    height_scale,
                    height_offset,
                    width_scale,
                    width_offset,
                    input_height,
                    input_width,
                    h_nearest,
                    w_nearest);
            }

            // Check if coordinates are valid
            bool h_valid = is_coordinate_valid(h_nearest, input_height);
            bool w_valid = is_coordinate_valid(w_nearest, input_width);

            // Read data if valid, write zeros if invalid (padding_mode="zeros")
            if (h_valid && w_valid) {
                const uint32_t nearest_stick_index = batch_offset + (h_nearest * input_width) + w_nearest;
                const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(nearest_stick_index);
                noc_async_read(remote_noc_addr, l1_write_work_addr, input_stick_size);
            } else {
                // Write zeros for out-of-bounds coordinates
                volatile tt_l1_ptr uint16_t* zero_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_work_addr);
                const uint32_t stick_size_words = input_stick_size / sizeof(uint16_t);
                for (uint32_t i = 0; i < stick_size_words; ++i) {
                    zero_ptr[i] = 0;
                }
            }

            l1_write_work_addr += input_stick_size;
        }

        // Wait for all input reads to complete
        noc_async_read_barrier();

        // Now write to DRAM output directly from the work CB
        // No need for cb_push_back/cb_wait_front since same kernel writes and reads
        uint32_t l1_read_work_addr = get_write_ptr(work_cb_index);  // Use same address we wrote to
        for (uint32_t grid_idx = 0; grid_idx < grid_batching_factor; ++grid_idx) {
            const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(output_stick_id);
            noc_async_write(l1_read_work_addr, output_noc_addr, output_stick_size);
            l1_read_work_addr += output_stick_size;
            ++output_stick_id;
        }
        noc_async_write_barrier();

        // For split reader, advance output_stick_id to next grid point this reader will process
        // Reader 0 processes grids 0,2,4,... so after grid 0 it skips grid 1's output
        // Reader 1 processes grids 1,3,5,... so after grid 1 it skips grid 2's output
        if constexpr (enable_split_reader) {
            output_stick_id += grid_batching_factor;  // Skip the other reader's output batch
        }
    }
}
