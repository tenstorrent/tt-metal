// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "compile_time_args.h"
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
#include "../grid_sample_reader_common.hpp"

#define PRINT_AND_PROFILE 1
#if PRINT_AND_PROFILE
#include "debug/dprint.h"
#endif

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
    uint32_t l1_write_output_addr,
    uint32_t spatial_pos = 0) {
    // Compute scaling factors as constexpr (same as in common utilities)
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    int32_t nearest_h, nearest_w;

    if constexpr (use_precomputed_grid) {
        // For precomputed grid: read the precomputed coordinates directly
        // The grid should contain int16 coordinates stored as uint16 bit patterns
        uint16_t h_bits = grid_ptr[grid_idx * 2 + 0];
        uint16_t w_bits = grid_ptr[grid_idx * 2 + 1];
        nearest_h = static_cast<int32_t>(static_cast<int16_t>(h_bits));
        nearest_w = static_cast<int32_t>(static_cast<int16_t>(w_bits));
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
        noc_async_read_barrier();
    } else {
        for (uint32_t i = 0; i < input_stick_nbytes; i += sizeof(uint32_t)) {
            volatile tt_l1_ptr uint32_t* zero_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_output_addr + i);
            *zero_ptr = 0;
        }
    }
}

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t num_pages = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(1);  // output_cb_index like sharded
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t input_height = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t grid_batches = get_compile_time_arg_val(6);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(7);
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(8);
    constexpr bool use_precomputed_grid = get_compile_time_arg_val(9);
    constexpr bool align_corners = get_compile_time_arg_val(10);

    constexpr auto src_args = TensorAccessorArgs<11>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto grid_tensor_accessor = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    const uint32_t end_id = start_page_id + num_pages;

    // Calculate starting batch from starting spatial position (avoid division in loop)
    uint32_t curr_batch = start_page_id / output_hw_size;
    uint32_t spatial_points_processed = start_page_id % output_hw_size;
    uint32_t batch_offset = curr_batch * input_height * input_width;

    // Get output CB base address like sharded writer
    const uint32_t l1_write_output_base_addr = get_write_ptr(output_cb_index);

// Debug range information
#if PRINT_AND_PROFILE
    DPRINT << "CORE RANGE: start_page_id=" << start_page_id << " num_pages=" << num_pages << " end_id=" << end_id
           << " (exclusive)" << ENDL();
    DPRINT << "BATCH SETUP: output_hw_size=" << output_hw_size << " curr_batch=" << curr_batch
           << " batch_offset=" << batch_offset << ENDL();
#endif
    //  Outer loop: iterate over spatial positions (output sticks)
    for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
        // Read the grid stick for this spatial position (contains grid_batches sets of coordinates)
        uint32_t l1_write_grid_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = grid_tensor_accessor.get_noc_addr(spatial_pos);

        noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_nbytes);
        noc_async_read_barrier();

        // Wait for grid data to be available and get read pointer
        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_grid_addr);

        // Calculate output address for this spatial position (like sharded writer)
        const uint32_t output_stick_offset = (spatial_pos - start_page_id);
        uint32_t l1_write_output_addr = l1_write_output_base_addr + output_stick_offset * input_stick_nbytes;

        //  Inner loop: process grid_batches coordinate sets within this spatial position
        for (uint32_t grid_idx = 0; grid_idx < grid_batches; ++grid_idx) {
            // Calculate pixel offset within the output stick
            const uint32_t pixel_offset = grid_idx * input_stick_nbytes;
            const uint32_t final_output_addr = l1_write_output_addr + pixel_offset;

            // Process grid point for nearest neighbor - write to L1 output CB
            process_grid_point_nearest<
                grid_dtype,
                use_precomputed_grid,
                align_corners,
                input_height,
                input_width,
                input_stick_nbytes,
                output_cb_index>(
                grid_ptr, grid_idx, input_tensor_accessor, batch_offset, final_output_addr, spatial_pos);
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
