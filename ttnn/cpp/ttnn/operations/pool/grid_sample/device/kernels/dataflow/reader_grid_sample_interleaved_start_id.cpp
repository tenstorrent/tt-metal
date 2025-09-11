// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/height_sharded_reader_common.hpp"

#define ALWI inline __attribute__((always_inline))

constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

// Data type constants (from ttnn/api/ttnn/tensor/types.hpp DataType enum)
constexpr uint32_t DTYPE_BFLOAT16 = 0;
constexpr uint32_t DTYPE_FLOAT32 = 1;

ALWI bool is_coordinate_valid(int32_t coord, uint32_t max_size) {
    return (coord >= 0) && (coord < static_cast<int32_t>(max_size));
}

ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

inline float bfloat16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

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
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(7);
    constexpr uint32_t grid_batches = get_compile_time_arg_val(8);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(9);

    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto s1 = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    const uint32_t end_id = start_page_id + num_pages;

    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);

    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;

    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

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

    // Outer loop: iterate over spatial positions (output sticks)
    for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
        // Read the grid stick for this spatial position (contains grid_batches sets of coordinates)
        uint32_t l1_write_grid_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = s0.get_noc_addr(spatial_pos);

        noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_nbytes);
        noc_async_read_barrier();

        // Cast to appropriate pointer type based on grid data type
        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_grid_addr);
        volatile tt_l1_ptr float* grid_float_ptr = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_grid_addr);

        uint32_t curr_batch = spatial_pos / output_hw_size;
        uint32_t batch_offset = curr_batch * input_height * input_width;

        // Inner loop: process grid_batches coordinate sets within this spatial position
        for (uint32_t grid_idx = 0; grid_idx < grid_batches; ++grid_idx) {
            uint16_t weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf;
            int32_t h0, h1, w0, w1;

#ifdef USE_PRECOMPUTED_GRID
            // Each precomputed grid entry has 6 values: h0, w0, weight_nw, weight_ne, weight_sw, weight_se
            uint32_t precomputed_data_offset = grid_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT;
            int16_t h0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 0]);
            int16_t w0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 1]);

            h0 = static_cast<int32_t>(h0_raw);
            w0 = static_cast<int32_t>(w0_raw);
            h1 = h0 + 1;
            w1 = w0 + 1;

            // Read precomputed weights
            weight_nw_bf = grid_ptr[precomputed_data_offset + 2];
            weight_ne_bf = grid_ptr[precomputed_data_offset + 3];
            weight_sw_bf = grid_ptr[precomputed_data_offset + 4];
            weight_se_bf = grid_ptr[precomputed_data_offset + 5];
#else
            // Each regular grid entry has 2 values: x, y coordinates
            float h_coord_rel, w_coord_rel;
            if constexpr (grid_dtype == DTYPE_FLOAT32) {
                // For FLOAT32 grid, each coordinate is a 32-bit float
                // Read from the base address with proper float32 offsets
                volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_grid_addr);
                uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                w_coord_rel = float_data[float_offset + 0];  // x coordinate
                h_coord_rel = float_data[float_offset + 1];  // y coordinate
            } else {
                // For BFLOAT16 grid, read as uint16 and convert
                uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
                uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
                h_coord_rel = bfloat16_to_float(h_coord_raw);
                w_coord_rel = bfloat16_to_float(w_coord_raw);
            }

            float h_coord_image = h_coord_rel * height_scale + height_offset;
            float w_coord_image = w_coord_rel * width_scale + width_offset;

            h0 = static_cast<int32_t>(floor(h_coord_image));
            h1 = h0 + 1;
            w0 = static_cast<int32_t>(floor(w_coord_image));
            w1 = w0 + 1;

            // Calculate bilinear interpolation weights
            float h0_f = static_cast<float>(h0);
            float w0_f = static_cast<float>(w0);

            float h_frac = h_coord_image - h0_f;
            float w_frac = w_coord_image - w0_f;
            float h_frac_inv = 1.0f - h_frac;
            float w_frac_inv = 1.0f - w_frac;

            // Need to declare boundary checks before using them
            bool h0_valid = is_coordinate_valid(h0, input_height);
            bool h1_valid = is_coordinate_valid(h1, input_height);
            bool w0_valid = is_coordinate_valid(w0, input_width);
            bool w1_valid = is_coordinate_valid(w1, input_width);

            float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;  // North-West
            float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;      // North-East
            float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;      // South-West
            float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;          // South-East

            weight_nw_bf = float_to_bfloat16(weight_nw);
            weight_ne_bf = float_to_bfloat16(weight_ne);
            weight_sw_bf = float_to_bfloat16(weight_sw);
            weight_se_bf = float_to_bfloat16(weight_se);
#endif

            // For precomputed grid, we need to compute boundary checks here
            // since they weren't computed in the #ifdef section above
#ifdef USE_PRECOMPUTED_GRID
            bool h0_valid = is_coordinate_valid(h0, input_height);
            bool h1_valid = is_coordinate_valid(h1, input_height);
            bool w0_valid = is_coordinate_valid(w0, input_width);
            bool w1_valid = is_coordinate_valid(w1, input_width);
#endif

            // Reserve CB space for 4 corner input sticks for this grid
            cb_reserve_back(input_cb_index, 1);
            uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

            // Read 4 corner input sticks
            if (h0_valid && w0_valid) {
                uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
                uint64_t dram_read_addr = s1.get_noc_addr(north_west_stick_index);
                noc_async_read(dram_read_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h0_valid && w1_valid) {
                uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
                uint64_t dram_read_addr = s1.get_noc_addr(north_east_stick_index);
                noc_async_read(dram_read_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h1_valid && w0_valid) {
                uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
                uint64_t dram_read_addr = s1.get_noc_addr(south_west_stick_index);
                noc_async_read(dram_read_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h1_valid && w1_valid) {
                uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;
                uint64_t dram_read_addr = s1.get_noc_addr(south_east_stick_index);
                noc_async_read(dram_read_addr, l1_write_input_addr, input_stick_nbytes);
            }

            // Store bilinear interpolation weights for this grid
            cb_reserve_back(scalar_cb_index, 1);
            fill_four_val(get_write_ptr(scalar_cb_index), weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
            cb_push_back(scalar_cb_index, 1);

            noc_async_read_barrier();
            cb_push_back(input_cb_index, 1);
        }
    }
}
