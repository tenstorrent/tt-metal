// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t scale_d = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_d = get_compile_time_arg_val(5);
    constexpr uint32_t out_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_w = get_compile_time_arg_val(7);
    constexpr uint32_t in_d = get_compile_time_arg_val(8);
    constexpr uint32_t in_h = get_compile_time_arg_val(9);
    constexpr uint32_t in_w = get_compile_time_arg_val(10);

    // Get TensorAccessor args for output buffer
    constexpr auto dst_args = TensorAccessorArgs<11>();

    // Runtime arguments
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Create TensorAccessor for output buffer
    const auto accessor = TensorAccessor(dst_args, dst_addr, stick_size_bytes);

    // Process each input stick
    for (uint32_t i = 0; i < num_sticks; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_addr = get_read_ptr(cb_id);

        // Calculate 5D position from linear stick index
        // stick_idx represents position in flattened (N, D, H, W) space
        uint32_t stick_idx = start_stick_id + i;

        // Decompose linear index to 5D coordinates
        // Layout: N * (D*H*W) + D * (H*W) + H * W + W
        uint32_t spatial_size = in_d * in_h * in_w;
        uint32_t curr_batch = stick_idx / spatial_size;
        uint32_t curr_idx = stick_idx % spatial_size;

        uint32_t curr_w = curr_idx % in_w;
        uint32_t remainder = curr_idx / in_w;
        uint32_t curr_h = remainder % in_h;
        uint32_t curr_d = remainder / in_h;

        // Calculate base output position
        uint32_t out_d_base = curr_d * scale_d;
        uint32_t out_h_base = curr_h * scale_h;
        uint32_t out_w_base = curr_w * scale_w;

        // Write to all upsampled positions
        for (uint32_t d = 0; d < scale_d; d++) {
            for (uint32_t h = 0; h < scale_h; h++) {
                for (uint32_t w = 0; w < scale_w; w++) {
                    // Calculate output stick index
                    // Output layout: N * (out_d*out_h*out_w) + out_d * (out_h*out_w) + out_h * out_w + out_w
                    uint32_t out_stick_idx = curr_batch * (out_d * out_h * out_w) + (out_d_base + d) * (out_h * out_w) +
                                             (out_h_base + h) * out_w + (out_w_base + w);

                    // Get output NOC address using TensorAccessor
                    uint64_t dst_noc_addr = accessor.get_noc_addr(out_stick_idx);

                    // Write the stick to output location
                    noc_async_write(l1_addr, dst_noc_addr, stick_size_bytes);
                }
            }
        }

        cb_pop_front(cb_id, 1);
    }

    // Wait for all writes to complete
    noc_async_write_barrier();
}
