// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_input_pages = get_arg_val<uint32_t>(1);
    uint32_t page_size = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_page_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t scale_d = get_compile_time_arg_val(4);
    constexpr uint32_t scale_h = get_compile_time_arg_val(5);
    constexpr uint32_t scale_w = get_compile_time_arg_val(6);
    constexpr uint32_t input_d = get_compile_time_arg_val(7);
    constexpr uint32_t input_h = get_compile_time_arg_val(8);
    constexpr uint32_t input_w = get_compile_time_arg_val(9);

    const auto s0 =
        get_interleaved_addr_gen<dst0_is_dram, dst_page_size_is_pow2>(dst_addr, page_size, dst_log_base_2_of_page_size);

    constexpr uint32_t output_d = input_d * scale_d;
    constexpr uint32_t output_h = input_h * scale_h;
    constexpr uint32_t output_w = input_w * scale_w;

    // Process each input page
    for (uint32_t input_page = 0; input_page < num_input_pages; ++input_page) {
        cb_wait_front(cb_id_out0, 1);
        uint64_t l1_read_addr = get_read_ptr(cb_id_out0);

        // Calculate input coordinates from linear page index
        // For 5D tensor [N, D, H, W, C], pages are organized as N*D*H*W pages
        uint32_t n = input_page / (input_d * input_h * input_w);
        uint32_t remaining = input_page % (input_d * input_h * input_w);
        uint32_t d = remaining / (input_h * input_w);
        remaining = remaining % (input_h * input_w);
        uint32_t h = remaining / input_w;
        uint32_t w = remaining % input_w;

        // Calculate base output page index for this input position
        uint32_t output_base = n * (output_d * output_h * output_w) + (d * scale_d) * (output_h * output_w) +
                               (h * scale_h) * output_w + (w * scale_w);

        // Replicate this input page to all corresponding output positions
        for (uint32_t dd = 0; dd < scale_d; ++dd) {
            for (uint32_t hh = 0; hh < scale_h; ++hh) {
                for (uint32_t ww = 0; ww < scale_w; ++ww) {
                    uint32_t output_offset = dd * (output_h * output_w) + hh * output_w + ww;
                    uint32_t output_page = output_base + output_offset;

                    uint64_t dst_noc_addr = get_noc_addr(output_page, s0);
                    noc_async_write(l1_read_addr, dst_noc_addr, page_size);
                }
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
