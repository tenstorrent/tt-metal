// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp"
#include "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/fixed_point_q16.h"

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    fixed_point_t cos_angle = static_cast<fixed_point_t>(get_arg_val<uint32_t>(3));
    fixed_point_t sin_angle = static_cast<fixed_point_t>(get_arg_val<uint32_t>(4));
    fixed_point_t center_x = static_cast<fixed_point_t>(get_arg_val<uint32_t>(5));
    fixed_point_t center_y = static_cast<fixed_point_t>(get_arg_val<uint32_t>(6));
    uint32_t fill_value_bf16 = get_arg_val<uint32_t>(7);

    // Compile-time arguments
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t input_batch = get_compile_time_arg_val(2);
    constexpr uint32_t input_height = get_compile_time_arg_val(3);
    constexpr uint32_t input_width = get_compile_time_arg_val(4);
    constexpr uint32_t input_channels = get_compile_time_arg_val(5);

    constexpr auto src_args = TensorAccessorArgs<6>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks; local_stick_idx++) {
        const uint32_t global_stick_idx = start_stick_id + local_stick_idx;

        cb_reserve_back(output_cb_index, 1);
        uint32_t l1_write_addr = get_write_ptr(output_cb_index);

        // Decode output pixel position from global stick index (NHWC format)
        const uint32_t batch_idx = global_stick_idx / (input_height * input_width);
        const uint32_t spatial_idx = global_stick_idx % (input_height * input_width);
        const uint32_t y_out = spatial_idx / input_width;
        const uint32_t x_out = spatial_idx % input_width;

        // Compute source coordinates using inverse rotation
        //  Translate to center-relative coordinates
        const fixed_point_t x_out_q16 = int_to_q16(x_out);
        const fixed_point_t y_out_q16 = int_to_q16(y_out);
        const fixed_point_t x_centered = q16_sub(x_out_q16, center_x);
        const fixed_point_t y_centered = q16_sub(y_out_q16, center_y);

        // Apply inverse rotation
        const fixed_point_t x_in = q16_mul_sub_add(x_centered, cos_angle, y_centered, sin_angle, center_x);
        const fixed_point_t y_in = q16_mul_add_add(x_centered, sin_angle, y_centered, cos_angle, center_y);

        // Round to nearest pixel
        const int32_t nearest_x = q16_to_int_round(x_in);
        const int32_t nearest_y = q16_to_int_round(y_in);

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

        cb_push_back(output_cb_index, 1);
    }
}
