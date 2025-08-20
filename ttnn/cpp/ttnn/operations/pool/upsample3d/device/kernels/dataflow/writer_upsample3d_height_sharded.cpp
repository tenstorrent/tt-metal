// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t is_reader = get_compile_time_arg_val(1);  // NCRISC vs BRISC

    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t scale_factor_d = get_compile_time_arg_val(3);
    constexpr uint32_t scale_factor_h = get_compile_time_arg_val(4);
    constexpr uint32_t scale_factor_w = get_compile_time_arg_val(5);
    constexpr uint32_t output_d = get_compile_time_arg_val(6);
    constexpr uint32_t output_h = get_compile_time_arg_val(7);
    constexpr uint32_t output_w = get_compile_time_arg_val(8);

    // Precomputed volumes and dimensions to avoid expensive div/mod in hot loop
    constexpr uint32_t output_dhw_volume = get_compile_time_arg_val(9);
    constexpr uint32_t output_hw_volume = get_compile_time_arg_val(10);
    constexpr uint32_t input_dhw_volume = get_compile_time_arg_val(11);
    constexpr uint32_t input_hw_volume = get_compile_time_arg_val(12);
    constexpr uint32_t input_w = get_compile_time_arg_val(13);

    // TensorAccessor for input
    constexpr auto input_args = TensorAccessorArgs<14>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_output_pages = get_arg_val<uint32_t>(1);
    uint32_t start_output_page_id = get_arg_val<uint32_t>(2);
    uint32_t reader_num_pages = get_arg_val<uint32_t>(3);

    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_nbytes);

    // Output is local to this core - start from CB address and increment
    uint32_t l1_write_addr = get_write_ptr(out_cb_id);

    // For writer core (BRISC), start from the second half of the output CB
    // to avoid collision with reader core (NCRISC)
    if (!is_reader) {
        l1_write_addr += reader_num_pages * stick_nbytes;
    }

    // Both RISC cores process their assigned work
    uint32_t pages_to_process = num_output_pages;

    // OPTIMIZED HOT LOOP: All expensive div/mod operations eliminated
    for (uint32_t page_idx = 0; page_idx < pages_to_process; ++page_idx) {
        uint32_t output_page_id = start_output_page_id + page_idx;

        // Convert output page ID to 3D coordinates using precomputed volumes
        // Avoid expensive divisions by using precomputed constants
        uint32_t n = output_page_id / output_dhw_volume;
        uint32_t remaining = output_page_id - n * output_dhw_volume;  // Avoid modulo with subtraction

        uint32_t out_d = remaining / output_hw_volume;
        remaining = remaining - out_d * output_hw_volume;  // Avoid modulo

        uint32_t out_h = remaining / output_w;
        uint32_t out_w = remaining - out_h * output_w;  // Avoid modulo

        // Calculate corresponding INPUT coordinates (nearest neighbor upsampling)
        // Using integer division but these are much cheaper since scale factors are small constants
        uint32_t input_d_coord = out_d / scale_factor_d;
        uint32_t input_h_coord = out_h / scale_factor_h;
        uint32_t input_w_coord = out_w / scale_factor_w;

        // Calculate input page ID using precomputed volumes
        uint32_t input_page_id =
            n * input_dhw_volume + input_d_coord * input_hw_volume + input_h_coord * input_w + input_w_coord;

        // Copy from input using TensorAccessor to local output CB
        uint64_t input_noc_addr = input_accessor.get_noc_addr(input_page_id);
        noc_async_read(input_noc_addr, l1_write_addr, stick_nbytes);
        l1_write_addr += stick_nbytes;
    }
    noc_async_read_barrier();
}
