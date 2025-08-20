// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Compile-time arguments (no config tensor!)
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(2);  // NCRISC vs BRISC

    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t scale_factor_d = get_compile_time_arg_val(4);
    constexpr uint32_t scale_factor_h = get_compile_time_arg_val(5);
    constexpr uint32_t scale_factor_w = get_compile_time_arg_val(6);
    constexpr uint32_t output_d = get_compile_time_arg_val(7);
    constexpr uint32_t output_h = get_compile_time_arg_val(8);
    constexpr uint32_t output_w = get_compile_time_arg_val(9);

    // TensorAccessor setup like the working interleaved kernels
    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t num_output_pages = get_arg_val<uint32_t>(2);
    uint32_t start_output_page_id = get_arg_val<uint32_t>(3);

    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_nbytes);
    const auto output_accessor = TensorAccessor(output_args, output_addr, stick_nbytes);

    // Calculate input dimensions
    const uint32_t input_d = output_d / scale_factor_d;
    const uint32_t input_h = output_h / scale_factor_h;
    const uint32_t input_w = output_w / scale_factor_w;

    // Both RISC cores process their assigned work
    uint32_t pages_to_process = num_output_pages;
    uint32_t start_page = start_output_page_id;

    // WORK FROM OUTPUT PERSPECTIVE: Process assigned output pages
    for (uint32_t page_idx = 0; page_idx < pages_to_process; ++page_idx) {
        uint32_t output_page_id = start_page + page_idx;

        // Convert output page ID to 3D coordinates (n, d, h, w)
        // For tensor [N, D, H, W, C]: page_id = n*(D*H*W) + d*(H*W) + h*W + w
        uint32_t output_dhw_volume = output_d * output_h * output_w;
        uint32_t n = output_page_id / output_dhw_volume;
        uint32_t remaining = output_page_id % output_dhw_volume;

        uint32_t output_hw_volume = output_h * output_w;
        uint32_t out_d = remaining / output_hw_volume;
        remaining = remaining % output_hw_volume;

        uint32_t out_h = remaining / output_w;
        uint32_t out_w = remaining % output_w;

        // Calculate corresponding INPUT coordinates (nearest neighbor upsampling)
        uint32_t input_d_coord = out_d / scale_factor_d;  // Integer division for nearest neighbor
        uint32_t input_h_coord = out_h / scale_factor_h;
        uint32_t input_w_coord = out_w / scale_factor_w;

        // Calculate input page ID from input coordinates
        uint32_t input_dhw_volume = input_d * input_h * input_w;
        uint32_t input_hw_volume = input_h * input_w;
        uint32_t input_page_id =
            n * input_dhw_volume + input_d_coord * input_hw_volume + input_h_coord * input_w + input_w_coord;

        // Direct copy from input NOC address to local output address
        uint64_t input_noc_addr = input_accessor.get_noc_addr(input_page_id);
        uint32_t local_output_addr = output_accessor.get_noc_addr(output_page_id);

        // Copy input directly to local output address
        noc_async_read(input_noc_addr, local_output_addr, stick_nbytes);
    }
    noc_async_read_barrier();
}
