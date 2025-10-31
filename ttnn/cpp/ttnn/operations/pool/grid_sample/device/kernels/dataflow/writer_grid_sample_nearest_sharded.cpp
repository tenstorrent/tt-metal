// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
#include "../grid_sample_reader_common.hpp"

// Process single grid point for nearest neighbor - adapted from common utilities
template <
    uint32_t grid_dtype,
    bool use_precomputed_grid,
    bool align_corners,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    uint32_t output_cb_index,
    typename TensorAccessor,
    typename GridPtrType>
ALWI void process_grid_point_nearest(
    GridPtrType grid_ptr,
    uint32_t grid_idx,
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset,
    uint32_t l1_write_output_addr) {
    // Compute scaling factors as constexpr (same as in common utilities)
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    int32_t nearest_h, nearest_w;

    if constexpr (use_precomputed_grid) {
        // Each precomputed grid entry for nearest mode has 2 values (coordinates only)
        const uint32_t precomputed_data_offset = grid_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST;
        const int16_t h_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 0]);
        const int16_t w_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 1]);

        nearest_h = static_cast<int32_t>(h_raw);
        nearest_w = static_cast<int32_t>(w_raw);
    } else {
        // Each regular grid entry has 2 values: x, y coordinates - compute nearest neighbor
        float h_coord_rel, w_coord_rel;
        if constexpr (grid_dtype == DTYPE_FLOAT32) {
            // For FLOAT32 grid, each coordinate is a 32-bit float
            volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(grid_ptr);
            const uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            w_coord_rel = float_data[float_offset + 0];  // x coordinate
            h_coord_rel = float_data[float_offset + 1];  // y coordinate
        } else {
            // For BFLOAT16 grid, read as uint16 and convert
            const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            const uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
            const uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
            h_coord_rel = bfloat16_to_float(h_coord_raw);
            w_coord_rel = bfloat16_to_float(w_coord_raw);
        }

        const float h_coord_image = h_coord_rel * height_scale + height_offset;
        const float w_coord_image = w_coord_rel * width_scale + width_offset;
        if constexpr (align_corners) {
            // For align_corners=True, use floor(coord) directly
            nearest_h = static_cast<int32_t>(floor(h_coord_image));
            nearest_w = static_cast<int32_t>(floor(w_coord_image));
        } else {
            // For nearest neighbor, use floor(coord + 0.5) to match preprocessing
            nearest_h = static_cast<int32_t>(floor(h_coord_image + 0.5f));
            nearest_w = static_cast<int32_t>(floor(w_coord_image + 0.5f));
        }
    }

    // Boundary checks - optimized for precomputed grid with sentinel values
    bool h_valid, w_valid;
    if constexpr (use_precomputed_grid) {
        // For precomputed grid, check sentinel value (-1) for invalid coordinates
        h_valid = (nearest_h != -1);
        w_valid = (nearest_w != -1);
    } else {
        // For regular grid, do full coordinate validation
        h_valid = is_coordinate_valid(nearest_h, input_height);
        w_valid = is_coordinate_valid(nearest_w, input_width);
    }

    if (h_valid && w_valid) {
        // Read the nearest neighbor pixel
        const uint32_t input_stick_index = batch_offset + (nearest_h * input_width) + nearest_w;
        const uint64_t input_noc_addr = input_tensor_accessor.get_noc_addr(input_stick_index);
        noc_async_read(input_noc_addr, l1_write_output_addr, input_stick_nbytes);
    } else {
        // Out of bounds - fill with zeros
        for (uint32_t i = 0; i < input_stick_nbytes; i += sizeof(uint32_t)) {
            volatile tt_l1_ptr uint32_t* zero_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_output_addr + i);
            *zero_ptr = 0;
        }
    }
}

// Advance grid index utility - same as sharded reader
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
    // Runtime arguments - same as sharded reader
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t global_grid_stick_start = get_arg_val<uint32_t>(1);

    // Compile time arguments - same as sharded reader
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);  // output instead of scalar
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t grid_batching_factor = get_compile_time_arg_val(7);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t grid_hw = get_compile_time_arg_val(9);
    constexpr uint32_t use_precomputed_grid = get_compile_time_arg_val(10);
    constexpr uint32_t align_corners = get_compile_time_arg_val(11);
    constexpr uint32_t split_reader = get_compile_time_arg_val(12);
    constexpr uint32_t reader_id = get_compile_time_arg_val(13);
    constexpr uint32_t grid_nsticks_per_core = get_compile_time_arg_val(14);

    // Input tensor accessor for remote NOC reads - same as sharded reader
    constexpr auto input_tensor_args = TensorAccessorArgs<15>();
    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_addr, input_stick_nbytes);

    // Calculate starting batch from global grid stick position
    const uint32_t starting_batch = global_grid_stick_start / grid_hw;

    // Get local grid data base address (already in L1)
    const uint32_t l1_grid_base_addr = get_read_ptr(grid_cb_index);
    const uint32_t l1_write_output_base_addr = get_write_ptr(output_cb_index);

    // Process each grid stick assigned to this core
    uint32_t grid_stick_idx = 0;
    uint32_t l1_grid_addr = l1_grid_base_addr;

    // For split reader: track grid point index starting from reader_id
    uint32_t in_grid_row_idx = 0;

    // Track current batch and grid position for batch increment logic
    uint32_t curr_batch = starting_batch;
    uint32_t grid_points_processed = global_grid_stick_start % grid_hw;
    // Advance at start if needed (for split reader)
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

        // Reserve CB space for output
        // cb_reserve_back(output_cb_index, 1);
        uint32_t l1_write_output_addr = l1_write_output_base_addr + grid_stick_idx * input_stick_nbytes;

        // Process nearest neighbor sampling and write directly to output
        process_grid_point_nearest<
            grid_dtype,
            use_precomputed_grid,
            align_corners,
            input_height,
            input_width,
            input_stick_nbytes,
            output_cb_index>(
            grid_stick_ptr, in_grid_row_idx, input_tensor_accessor, batch_offset, l1_write_output_addr);

        // Push output to CB
        // cb_push_back(output_cb_index, 1);

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
    noc_async_read_barrier();
}
