// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/height_sharded_reader_common.hpp"
#include "debug/dprint.h"

#define ALWI inline __attribute__((always_inline))

constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

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
    // Runtime arguments
    const uint32_t input_addr = get_arg_val<uint32_t>(0);

    // Compile time arguments
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t grid_nsticks_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t grid_batching_factor = get_compile_time_arg_val(8);
    constexpr uint32_t use_precomputed_grid = get_compile_time_arg_val(9);

    // Input tensor accessor for remote NOC reads
    constexpr auto input_tensor_args = TensorAccessorArgs<10>();
    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_addr, input_stick_nbytes);

    // Grid coordinates scaling factors (for standard grid mode)
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    // Zero out input CB to handle invalid coordinates properly
    zero_out_tiles<input_cb_index>();

    // Get local grid data base address (already in L1)
    const uint32_t l1_grid_base_addr = get_read_ptr(grid_cb_index);

    // Process each grid stick assigned to this core
    uint32_t grid_stick_idx = 0;
    uint32_t l1_grid_addr = l1_grid_base_addr;

    while (grid_stick_idx < grid_nsticks_per_core) {
        volatile tt_l1_ptr uint16_t* const grid_stick_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_grid_addr);

        // Process each batched grid point within this stick
        uint32_t batch_idx = 0;
        while (batch_idx < grid_batching_factor) {
            uint16_t weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf;
            int32_t h0, h1, w0, w1;
            uint32_t curr_image_batch = 0;  // Will need to calculate based on global position
            uint32_t batch_offset = curr_image_batch * input_height * input_width;

#ifdef USE_PRECOMPUTED_GRID
            // Each precomputed grid entry has 6 values: h0, w0, weight_nw, weight_ne, weight_sw, weight_se
            const uint32_t precomputed_data_offset = batch_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT;
            const int16_t h0_raw = *reinterpret_cast<volatile int16_t*>(&grid_stick_ptr[precomputed_data_offset + 0]);
            const int16_t w0_raw = *reinterpret_cast<volatile int16_t*>(&grid_stick_ptr[precomputed_data_offset + 1]);

            h0 = static_cast<int32_t>(h0_raw);
            w0 = static_cast<int32_t>(w0_raw);
            h1 = h0 + 1;
            w1 = w0 + 1;

            // Read precomputed weights
            weight_nw_bf = grid_stick_ptr[precomputed_data_offset + 2];
            weight_ne_bf = grid_stick_ptr[precomputed_data_offset + 3];
            weight_sw_bf = grid_stick_ptr[precomputed_data_offset + 4];
            weight_se_bf = grid_stick_ptr[precomputed_data_offset + 5];
#else
            // Each regular grid entry has 2 values: x, y coordinates
            const uint32_t coordinate_pair_offset = batch_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            const uint16_t h_coord_raw = grid_stick_ptr[coordinate_pair_offset + 1];  // y coordinate
            const uint16_t w_coord_raw = grid_stick_ptr[coordinate_pair_offset + 0];  // x coordinate

            const float h_coord_rel = bfloat16_to_float(h_coord_raw);
            const float w_coord_rel = bfloat16_to_float(w_coord_raw);

            const float h_coord_image = h_coord_rel * height_scale + height_offset;
            const float w_coord_image = w_coord_rel * width_scale + width_offset;

            h0 = static_cast<int32_t>(floor(h_coord_image));
            h1 = h0 + 1;
            w0 = static_cast<int32_t>(floor(w_coord_image));
            w1 = w0 + 1;

            // Calculate bilinear interpolation weights
            const float h0_f = static_cast<float>(h0);
            const float w0_f = static_cast<float>(w0);

            const float h_frac = h_coord_image - h0_f;
            const float w_frac = w_coord_image - w0_f;
            const float h_frac_inv = 1.0f - h_frac;
            const float w_frac_inv = 1.0f - w_frac;

            // Boundary checks
            const bool h0_valid = is_coordinate_valid(h0, input_height);
            const bool h1_valid = is_coordinate_valid(h1, input_height);
            const bool w0_valid = is_coordinate_valid(w0, input_width);
            const bool w1_valid = is_coordinate_valid(w1, input_width);

            const float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;  // North-West
            const float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;      // North-East
            const float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;      // South-West
            const float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;          // South-East

            weight_nw_bf = float_to_bfloat16(weight_nw);
            weight_ne_bf = float_to_bfloat16(weight_ne);
            weight_sw_bf = float_to_bfloat16(weight_sw);
            weight_se_bf = float_to_bfloat16(weight_se);
#endif

            // For precomputed grid, compute boundary checks here
#ifdef USE_PRECOMPUTED_GRID
            const bool h0_valid = is_coordinate_valid(h0, input_height);
            const bool h1_valid = is_coordinate_valid(h1, input_height);
            const bool w0_valid = is_coordinate_valid(w0, input_width);
            const bool w1_valid = is_coordinate_valid(w1, input_width);
#endif

            // Reserve CB space for 4 corner input sticks for this grid point

            cb_reserve_back(input_cb_index, 1);

            uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

            // Read 4 corner input sticks via NOC from remote input tensor shards
            if (h0_valid && w0_valid) {
                const uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
                const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(north_west_stick_index);
                noc_async_read(remote_noc_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h0_valid && w1_valid) {
                const uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
                const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(north_east_stick_index);
                noc_async_read(remote_noc_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h1_valid && w0_valid) {
                const uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
                const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(south_west_stick_index);
                noc_async_read(remote_noc_addr, l1_write_input_addr, input_stick_nbytes);
            }
            l1_write_input_addr += input_stick_nbytes;

            if (h1_valid && w1_valid) {
                const uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;
                const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(south_east_stick_index);
                noc_async_read(remote_noc_addr, l1_write_input_addr, input_stick_nbytes);
            }

            noc_async_read_barrier();
            cb_push_back(input_cb_index, 1);

            // Write scalar weights to scalar CB
            cb_reserve_back(scalar_cb_index, 1);
            const uint32_t l1_write_scalar_addr = get_write_ptr(scalar_cb_index);
            fill_four_val(l1_write_scalar_addr, weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
            cb_push_back(scalar_cb_index, 1);

            ++batch_idx;
            if (batch_idx == grid_batching_factor) {
                l1_grid_addr += grid_stick_nbytes;
                ++grid_stick_idx;
            }
        }
    }
}
