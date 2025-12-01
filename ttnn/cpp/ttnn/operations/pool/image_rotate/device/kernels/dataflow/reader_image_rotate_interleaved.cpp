// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Include conv common for zero_out_tiles
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
// Include grid_sample helper functions for bilinear interpolation
#include "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp"
// Include Q16.16 fixed-point arithmetic helpers
#include "ttnn/cpp/ttnn/operations/pool/image_rotate/device/fixed_point_q16.hpp"

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

    // Rotation parameters are passed as Q16.16 fixed-point (pre-converted on host)
    const int32_t cos_angle_q16 = static_cast<int32_t>(cos_angle_bits);
    const int32_t sin_angle_q16 = static_cast<int32_t>(sin_angle_bits);
    const int32_t center_x_q16 = static_cast<int32_t>(center_x_bits);
    const int32_t center_y_q16 = static_cast<int32_t>(center_y_bits);

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
        // Convert output coordinates to Q16.16 fixed-point
        const int32_t x_out_q16 = int_to_q16(static_cast<int32_t>(x_out));
        const int32_t y_out_q16 = int_to_q16(static_cast<int32_t>(y_out));

        // Center the coordinates
        const int32_t x_centered_q16 = q16_sub(x_out_q16, center_x_q16);
        const int32_t y_centered_q16 = q16_sub(y_out_q16, center_y_q16);

        // Inverse rotation transformation using Q16.16 arithmetic
        // x_in = x_centered * cos_angle - y_centered * sin_angle + center_x
        const int32_t term1 = q16_mul(x_centered_q16, cos_angle_q16);
        const int32_t term2 = q16_mul(y_centered_q16, sin_angle_q16);
        const int32_t x_in_q16 = q16_add(q16_sub(term1, term2), center_x_q16);

        // y_in = x_centered * sin_angle + y_centered * cos_angle + center_y
        const int32_t term3 = q16_mul(x_centered_q16, sin_angle_q16);
        const int32_t term4 = q16_mul(y_centered_q16, cos_angle_q16);
        const int32_t y_in_q16 = q16_add(q16_add(term3, term4), center_y_q16);

        // Compute corner coordinates for bilinear interpolation
        // COORDINATE FIX: Use y_in for height, x_in for width (matching grid_sample convention)
        // Convert Q16.16 to plain int32 coordinates (floor operation)
        const int32_t h0 = q16_to_int(y_in_q16);
        const int32_t h1 = h0 + 1;
        const int32_t w0 = q16_to_int(x_in_q16);
        const int32_t w1 = w0 + 1;

        // Calculate bilinear interpolation weights using Q16.16 arithmetic
        // Extract fractional parts (already in Q16.16 format, range [0, 1))
        const int32_t h_frac_q16 = q16_frac(y_in_q16);
        const int32_t w_frac_q16 = q16_frac(x_in_q16);

        // Compute (1 - frac) for inverse fractions
        const int32_t h_frac_inv_q16 = q16_one_minus(h_frac_q16);
        const int32_t w_frac_inv_q16 = q16_one_minus(w_frac_q16);

        // Boundary checks
        const bool h0_valid = is_coordinate_valid(h0, input_height);
        const bool h1_valid = is_coordinate_valid(h1, input_height);
        const bool w0_valid = is_coordinate_valid(w0, input_width);
        const bool w1_valid = is_coordinate_valid(w1, input_width);

        // Compute bilinear interpolation weights in Q16.16
        const int32_t weight_nw_q16 = (h0_valid && w0_valid) ? q16_mul(h_frac_inv_q16, w_frac_inv_q16) : 0;
        const int32_t weight_ne_q16 = (h0_valid && w1_valid) ? q16_mul(h_frac_inv_q16, w_frac_q16) : 0;
        const int32_t weight_sw_q16 = (h1_valid && w0_valid) ? q16_mul(h_frac_q16, w_frac_inv_q16) : 0;
        const int32_t weight_se_q16 = (h1_valid && w1_valid) ? q16_mul(h_frac_q16, w_frac_q16) : 0;

        // Convert Q16.16 weights to float, then to bfloat16
        const uint16_t weight_nw_bf = float_to_bfloat16(q16_to_float(weight_nw_q16));
        const uint16_t weight_ne_bf = float_to_bfloat16(q16_to_float(weight_ne_q16));
        const uint16_t weight_sw_bf = float_to_bfloat16(q16_to_float(weight_sw_q16));
        const uint16_t weight_se_bf = float_to_bfloat16(q16_to_float(weight_se_q16));

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
