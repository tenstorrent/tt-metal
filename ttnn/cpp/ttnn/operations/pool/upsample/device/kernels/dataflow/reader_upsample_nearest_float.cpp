// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Runtime arguments
    const auto num_sticks = get_arg(args::num_sticks);          // Number of output sticks for this core
    const auto start_stick_id = get_arg(args::start_stick_id);  // Starting output stick ID

    // Compile-time arguments
    constexpr auto aligned_stick_nbytes = get_arg(args::aligned_input_page_size);  // Aligned stick size in bytes
    constexpr auto input_height = get_arg(args::input_height);                     // Input H
    constexpr auto input_width = get_arg(args::input_width);                       // Input W
    constexpr auto output_height = get_arg(args::output_height);                   // Output H
    constexpr auto output_width = get_arg(args::output_width);                     // Output W
    constexpr auto num_pages_per_width = get_arg(args::num_pages_across_width);    // Number of pages across width
    constexpr int32_t reciprocal_scale_h_fixed = get_arg(args::reciprocal_scale_h_fixed);  // input_h/output_h Q16.16
    constexpr int32_t reciprocal_scale_w_fixed = get_arg(args::reciprocal_scale_w_fixed);  // input_w/output_w Q16.16

    const auto input_tensor_accessor = TensorAccessor(tensor::input);

    DataflowBuffer out_dfb(dfb::out);
    Noc noc;

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
        out_dfb.reserve_back(1);

        // Read source stick from DRAM
        noc.async_read(input_tensor_accessor, out_dfb, aligned_stick_nbytes, {.page_id = src_stick_id}, {});

        // Wait for read to complete
        noc.async_read_barrier();

        // Push to CB
        out_dfb.push_back(1);

        page_id++;
    }
}
