// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "fixed_point_arithmetic.hpp"

void kernel_main() {
    // Runtime arguments
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);  // Input tensor DRAM address
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);         // Number of output sticks for this core
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);     // Starting output stick ID

    // Compile-time arguments
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);                // Output CB ID
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(1);     // Aligned stick size in bytes
    constexpr uint32_t input_height = get_compile_time_arg_val(2);             // Input H
    constexpr uint32_t input_width = get_compile_time_arg_val(3);              // Input W
    constexpr uint32_t output_height = get_compile_time_arg_val(4);            // Output H
    constexpr uint32_t output_width = get_compile_time_arg_val(5);             // Output W
    constexpr uint32_t num_pages_per_width = get_compile_time_arg_val(6);      // Number of pages across width
    constexpr int32_t reciprocal_scale_h_fixed = get_compile_time_arg_val(7);  // input_h/output_h in Q16.16
    constexpr int32_t reciprocal_scale_w_fixed = get_compile_time_arg_val(8);  // input_w/output_w in Q16.16

    // Tensor accessor compile-time args start at index 9
    constexpr auto src_args = TensorAccessorArgs<9>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_buffer_addr, aligned_stick_nbytes);

    // Process sticks assigned to this core
    uint32_t page_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        // Compute output coordinates (batch, y, x) from flat stick index
        // stick_id = batch * output_height * output_width + y_out * output_width + x_out
        const uint32_t stick_id = page_id / num_pages_per_width;
        const uint32_t in_stick_offset = page_id % num_pages_per_width;
        const uint32_t batch = stick_id / (output_height * output_width);
        const uint32_t remainder = stick_id % (output_height * output_width);
        const uint32_t y_out = remainder / output_width;
        const uint32_t x_out = remainder % output_width;

        // Map output coordinates to input coordinates using fixed-point arithmetic
        // src_y = floor(y_out * reciprocal_scale_h) = floor(y_out / scale_h)
        // src_x = floor(x_out * reciprocal_scale_w) = floor(x_out / scale_w)
        const uint32_t src_y =
            static_cast<uint32_t>(fixed_point_arithmetic::fixed_mul(y_out, reciprocal_scale_h_fixed));
        const uint32_t src_x =
            static_cast<uint32_t>(fixed_point_arithmetic::fixed_mul(x_out, reciprocal_scale_w_fixed));

        // Clamp source coordinates to valid range
        const uint32_t clamped_src_y = (src_y < input_height) ? src_y : (input_height - 1);
        const uint32_t clamped_src_x = (src_x < input_width) ? src_x : (input_width - 1);

        // Compute flat source stick index
        const uint32_t src_stick_id = batch * input_height * input_width * num_pages_per_width +
                                      clamped_src_y * input_width * num_pages_per_width +
                                      clamped_src_x * num_pages_per_width + in_stick_offset;

        // Reserve space in output CB
        cb_reserve_back(cb_id_out, 1);

        // Get L1 write address
        uint32_t l1_write_addr = get_write_ptr(cb_id_out);

        // Read source stick from DRAM
        uint64_t src_noc_addr = input_tensor_accessor.get_noc_addr(src_stick_id);
        noc_async_read(src_noc_addr, l1_write_addr, aligned_stick_nbytes);

        // Wait for read to complete
        noc_async_read_barrier();

        // Push to CB
        cb_push_back(cb_id_out, 1);

        page_id++;
    }
}
