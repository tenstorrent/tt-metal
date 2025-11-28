// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"

// Include conv common for zero_out_tiles
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
// Include grid_sample helper functions for bilinear interpolation
#include "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp"

#define ALWI inline __attribute__((always_inline))

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t cos_angle_bits = get_arg_val<uint32_t>(3);
    uint32_t sin_angle_bits = get_arg_val<uint32_t>(4);
    uint32_t center_x_bits = get_arg_val<uint32_t>(5);
    uint32_t center_y_bits = get_arg_val<uint32_t>(6);
    uint32_t fill_value_bf16 = get_arg_val<uint32_t>(7);

    // Compile-time arguments
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t input_batch = get_compile_time_arg_val(3);
    constexpr uint32_t input_height = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);

    // Reinterpret rotation parameter bits as float
    union {
        uint32_t u;
        float f;
    } cos_conv, sin_conv, cx_conv, cy_conv;

    cos_conv.u = cos_angle_bits;
    sin_conv.u = sin_angle_bits;
    cx_conv.u = center_x_bits;
    cy_conv.u = center_y_bits;

    const float cos_angle = cos_conv.f;
    const float sin_angle = sin_conv.f;
    const float center_x = cx_conv.f;
    const float center_y = cy_conv.f;

    // Tensor accessor for input tensor (starts at compile-time arg index 6)
    constexpr auto src_args = TensorAccessorArgs<6>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    // Precompute constants
    constexpr uint32_t hw_size = input_height * input_width;

    const uint32_t end_stick_id = start_stick_id + num_sticks;

    // Zero out the input CB at the start to handle out-of-bounds corners
    // (same pattern as grid_sample)
    zero_out_tiles<input_cb_index>();

    // Calculate starting batch from starting spatial position
    uint32_t curr_batch = start_stick_id / hw_size;
    uint32_t spatial_pos_in_batch = start_stick_id % hw_size;
    uint32_t batch_offset = curr_batch * hw_size;

    // Process each output pixel
    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; ++stick_id) {
        // Compute output pixel position (y_out, x_out) from linear stick index
        const uint32_t y_out = spatial_pos_in_batch / input_width;
        const uint32_t x_out = spatial_pos_in_batch % input_width;

        // Compute rotation: find source coordinates (x_in, y_in) for this output pixel
        // Using inverse rotation to find where each output pixel samples from
        const float x_centered = static_cast<float>(x_out) - center_x;
        const float y_centered = static_cast<float>(y_out) - center_y;

        // Inverse rotation transformation
        const float x_in = x_centered * cos_angle - y_centered * sin_angle + center_x;
        const float y_in = x_centered * sin_angle + y_centered * cos_angle + center_y;

        // Compute corner coordinates for bilinear interpolation
        // COORDINATE FIX: Use y_in for height, x_in for width (matching grid_sample convention)
        const int32_t h0 = static_cast<int32_t>(floor(y_in));
        const int32_t h1 = h0 + 1;
        const int32_t w0 = static_cast<int32_t>(floor(x_in));
        const int32_t w1 = w0 + 1;

        // Calculate bilinear interpolation weights
        const float h0_f = static_cast<float>(h0);
        const float w0_f = static_cast<float>(w0);

        const float h_frac = y_in - h0_f;
        const float w_frac = x_in - w0_f;
        const float h_frac_inv = 1.0f - h_frac;
        const float w_frac_inv = 1.0f - w_frac;

        // Boundary checks
        const bool h0_valid = is_coordinate_valid(h0, input_height);
        const bool h1_valid = is_coordinate_valid(h1, input_height);
        const bool w0_valid = is_coordinate_valid(w0, input_width);
        const bool w1_valid = is_coordinate_valid(w1, input_width);

        // Compute weights, zeroing out invalid corners
        const float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;
        const float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;
        const float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;
        const float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;

        // Convert weights to bfloat16
        const uint16_t weight_nw_bf = float_to_bfloat16(weight_nw);
        const uint16_t weight_ne_bf = float_to_bfloat16(weight_ne);
        const uint16_t weight_sw_bf = float_to_bfloat16(weight_sw);
        const uint16_t weight_se_bf = float_to_bfloat16(weight_se);

        // Reserve CB space for 4 corner input sticks
        cb_reserve_back(input_cb_index, 1);
        const uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

        // Read 4 corner input sticks using the common helper function
        read_four_corner_inputs(
            input_tensor_accessor,
            batch_offset,
            input_width,
            input_stick_nbytes,
            h0,
            h1,
            w0,
            w1,
            input_height,
            l1_write_input_addr);

        // Store bilinear interpolation weights
        cb_reserve_back(scalar_cb_index, 1);
        const uint32_t l1_write_scalar_addr = get_write_ptr(scalar_cb_index);
        fill_four_val(l1_write_scalar_addr, weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
        cb_push_back(scalar_cb_index, 1);

        // Wait for NOC reads to complete and push input CB
        noc_async_read_barrier();
        cb_push_back(input_cb_index, 1);

        // Update batch tracking
        ++spatial_pos_in_batch;
        if (spatial_pos_in_batch == hw_size) {
            spatial_pos_in_batch = 0;
            ++curr_batch;
            batch_offset = curr_batch * hw_size;
        }
    }
}
