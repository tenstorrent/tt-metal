// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/height_sharded_reader_common.hpp"
#include "../grid_sample_reader_common.hpp"

#define PRINT_AND_PROFILE 0
#if PRINT_AND_PROFILE
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"
#endif

ALWI void advance_grid_index(
    uint32_t& in_grid_row_idx,
    uint32_t& grid_stick_idx,
    uint32_t& l1_grid_addr,
    uint32_t& grid_points_processed,
    uint32_t& curr_batch,
    const uint32_t grid_batching_factor,
    const uint32_t grid_stick_nbytes,
    const uint32_t grid_hw,
    const uint32_t grid_nsticks_per_core) {
    ++in_grid_row_idx;
    if (in_grid_row_idx == grid_batching_factor) {
        in_grid_row_idx = 0;
        ++grid_stick_idx;
        l1_grid_addr += grid_stick_nbytes;
        ++grid_points_processed;
        if (grid_points_processed == grid_hw) {
            grid_points_processed = 0;
            ++curr_batch;
        }
    }
}

void kernel_main() {
    // Runtime arguments
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t global_grid_stick_start = get_arg_val<uint32_t>(1);

    // Compile time arguments
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t grid_batching_factor = get_compile_time_arg_val(7);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t grid_hw = get_compile_time_arg_val(9);
    constexpr uint32_t use_precomputed_grid = get_compile_time_arg_val(10);
    constexpr uint32_t split_reader = get_compile_time_arg_val(11);
    constexpr uint32_t reader_id = get_compile_time_arg_val(12);
    constexpr uint32_t grid_nsticks_per_core = get_compile_time_arg_val(13);

    // Input tensor accessor for remote NOC reads (updated for new arg count)
    constexpr auto input_tensor_args = TensorAccessorArgs<14>();
    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_addr, input_stick_nbytes);

    // Calculate starting batch from global grid stick position
    // All grid points in one grid stick are in the same batch
    const uint32_t starting_batch = global_grid_stick_start / grid_hw;

    // Zero out input CB to handle invalid coordinates properly
    zero_out_tiles<input_cb_index>();

    // Get local grid data base address (already in L1)
    const uint32_t l1_grid_base_addr = get_read_ptr(grid_cb_index);

    // Process each grid stick assigned to this core
    uint32_t grid_stick_idx = 0;
    uint32_t l1_grid_addr = l1_grid_base_addr;

    // For split reader: track grid point index starting from reader_id
    uint32_t in_grid_row_idx = 0;

    // Track current batch and grid position for batch increment logic
    uint32_t curr_batch = starting_batch;
    uint32_t grid_points_processed = global_grid_stick_start % grid_hw;

    // Advance at start if needed
    if constexpr (split_reader && reader_id == 1) {
        advance_grid_index(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw,
            grid_nsticks_per_core);
    }

    while (grid_stick_idx < grid_nsticks_per_core) {
        volatile tt_l1_ptr uint16_t* const grid_stick_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_grid_addr);

        uint32_t batch_offset = curr_batch * input_height * input_width;
        process_grid_point<
            grid_dtype,
            use_precomputed_grid,
            input_height,
            input_width,
            input_stick_nbytes,
            input_cb_index,
            scalar_cb_index>(grid_stick_ptr, in_grid_row_idx, input_tensor_accessor, batch_offset);

        // Always advance once after processing
        advance_grid_index(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw,
            grid_nsticks_per_core);

        // For split reader, advance one more time to skip the coordinate that the other reader will process
        if constexpr (split_reader) {
            advance_grid_index(
                in_grid_row_idx,
                grid_stick_idx,
                l1_grid_addr,
                grid_points_processed,
                curr_batch,
                grid_batching_factor,
                grid_stick_nbytes,
                grid_hw,
                grid_nsticks_per_core);
        }
    }
}
