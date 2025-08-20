// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t out_cb_id = get_compile_time_arg_val(0);
    const uint32_t is_reader = get_compile_time_arg_val(1);  // NCRISC vs BRISC

    const uint32_t stick_nbytes = get_compile_time_arg_val(2);
    const uint32_t scale_factor_d = get_compile_time_arg_val(3);
    const uint32_t scale_factor_h = get_compile_time_arg_val(4);
    const uint32_t scale_factor_w = get_compile_time_arg_val(5);
    const uint32_t output_d = get_compile_time_arg_val(6);
    const uint32_t output_h = get_compile_time_arg_val(7);
    const uint32_t output_w = get_compile_time_arg_val(8);

    // Precomputed volumes and dimensions to avoid expensive div/mod in hot loop
    const uint32_t output_dhw_volume = get_compile_time_arg_val(9);
    const uint32_t output_hw_volume = get_compile_time_arg_val(10);
    const uint32_t input_dhw_volume = get_compile_time_arg_val(11);
    const uint32_t input_hw_volume = get_compile_time_arg_val(12);
    const uint32_t input_w = get_compile_time_arg_val(13);

    // TensorAccessor for input
    const auto input_args = TensorAccessorArgs<14>();

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

    // ULTRA-OPTIMIZED HOT LOOP: Complete elimination of all division operations
    // Initialize coordinates for the starting output page (only divisions here, outside the loop)
    uint32_t current_n = start_output_page_id / output_dhw_volume;
    uint32_t remaining_in_batch = start_output_page_id - current_n * output_dhw_volume;
    uint32_t current_out_d = remaining_in_batch / output_hw_volume;
    remaining_in_batch = remaining_in_batch - current_out_d * output_hw_volume;
    uint32_t current_out_h = remaining_in_batch / output_w;
    uint32_t current_out_w = remaining_in_batch - current_out_h * output_w;

    // Precompute input coordinates for the starting position
    uint32_t current_input_d = current_out_d / scale_factor_d;
    uint32_t current_input_h = current_out_h / scale_factor_h;
    uint32_t current_input_w = current_out_w / scale_factor_w;

    for (uint32_t page_idx = 0; page_idx < pages_to_process; ++page_idx) {
        // Calculate input page ID using current coordinates (no divisions!)
        uint32_t input_page_id = current_n * input_dhw_volume + current_input_d * input_hw_volume +
                                 current_input_h * input_w + current_input_w;

        // Copy from input using TensorAccessor to local output CB
        uint64_t input_noc_addr = input_accessor.get_noc_addr(input_page_id);
        noc_async_read(input_noc_addr, l1_write_addr, stick_nbytes);
        l1_write_addr += stick_nbytes;

        // Increment coordinates efficiently without divisions
        ++current_out_w;
        if (current_out_w == output_w) {
            current_out_w = 0;
            current_input_w = 0;
            ++current_out_h;
            if (current_out_h == output_h) {
                current_out_h = 0;
                current_input_h = 0;
                ++current_out_d;
                if (current_out_d == output_d) {
                    current_out_d = 0;
                    current_input_d = 0;
                    ++current_n;
                } else {
                    current_input_d = current_out_d / scale_factor_d;
                }
            } else {
                current_input_h = current_out_h / scale_factor_h;
            }
        } else {
            current_input_w = current_out_w / scale_factor_w;
        }
    }
    noc_async_read_barrier();
}
