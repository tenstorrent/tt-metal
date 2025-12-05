// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
// Include for tensor accessor and common utilities
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"

// Helper function to check if coordinate is valid (in bounds)
inline bool is_coordinate_valid(int32_t coord, uint32_t max_val) {
    return coord >= 0 && coord < static_cast<int32_t>(max_val);
}

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    uint32_t cos_angle_bits = get_arg_val<uint32_t>(4);
    uint32_t sin_angle_bits = get_arg_val<uint32_t>(5);
    uint32_t center_x_bits = get_arg_val<uint32_t>(6);
    uint32_t center_y_bits = get_arg_val<uint32_t>(7);
    uint32_t fill_value_bf16 = get_arg_val<uint32_t>(8);

    // Compile-time arguments
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t output_stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t input_batch = get_compile_time_arg_val(3);
    constexpr uint32_t input_height = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t input_channels = get_compile_time_arg_val(6);
    constexpr uint32_t enable_split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);

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

    // Tensor accessor for input tensor (starts at compile-time arg index 9)
    constexpr auto src_args = TensorAccessorArgs<9>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    // Tensor accessor for output tensor
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_addr, output_stick_nbytes);

    // Initialize CB
    cb_reserve_back(output_cb_index, 1);
    uint32_t l1_write_addr = get_write_ptr(output_cb_index);

    // Process each output pixel
    // For split reader mode, each reader processes alternate pixels
    const uint32_t stride = enable_split_reader ? 2 : 1;
    const uint32_t initial_offset = enable_split_reader ? reader_id : 0;

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks; local_stick_idx += stride) {
        const uint32_t global_stick_idx = start_stick_id + local_stick_idx + initial_offset;

        // Skip if we've exceeded the total number of sticks
        if (global_stick_idx >= start_stick_id + num_sticks) {
            break;
        }

        // Decode output pixel position from global stick index (NHWC format)
        const uint32_t batch_idx = global_stick_idx / (input_height * input_width);
        const uint32_t spatial_idx = global_stick_idx % (input_height * input_width);
        const uint32_t y_out = spatial_idx / input_width;
        const uint32_t x_out = spatial_idx % input_width;

        // Compute source coordinates using inverse rotation
        // Translate to center-relative coordinates
        const float x_centered = static_cast<float>(x_out) - center_x;
        const float y_centered = static_cast<float>(y_out) - center_y;

        // Apply inverse rotation
        const float x_in = x_centered * cos_angle - y_centered * sin_angle + center_x;
        const float y_in = x_centered * sin_angle + y_centered * cos_angle + center_y;

        // Round to nearest pixel (nearest interpolation)
        const int32_t nearest_x = static_cast<int32_t>(round(x_in));
        const int32_t nearest_y = static_cast<int32_t>(round(y_in));

        // Check if the nearest pixel is in bounds
        const bool x_valid = is_coordinate_valid(nearest_x, input_width);
        const bool y_valid = is_coordinate_valid(nearest_y, input_height);

        if (x_valid && y_valid) {
            // Read the nearest neighbor pixel
            const uint32_t input_stick_index =
                batch_idx * (input_height * input_width) + nearest_y * input_width + nearest_x;
            const uint64_t input_noc_addr = input_tensor_accessor.get_noc_addr(input_stick_index);
            noc_async_read(input_noc_addr, l1_write_addr, input_stick_nbytes);
            noc_async_read_barrier();
        } else {
            // Fill the entire stick with the fill value
            volatile tt_l1_ptr uint16_t* output_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
            for (uint32_t i = 0; i < input_channels; i++) {
                output_ptr[i] = static_cast<uint16_t>(fill_value_bf16);
            }
        }

        // Write the output stick to DRAM
        cb_push_back(output_cb_index, 1);

        const uint32_t output_stick_index = global_stick_idx;
        const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(output_stick_index);

        cb_wait_front(output_cb_index, 1);
        uint32_t l1_read_addr = get_read_ptr(output_cb_index);
        noc_async_write(l1_read_addr, output_noc_addr, output_stick_nbytes);
        noc_async_write_barrier();
        cb_pop_front(output_cb_index, 1);
    }
}
