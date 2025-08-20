// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_size_bytes = get_compile_time_arg_val(1);

    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t t_out_start = get_arg_val<uint32_t>(1);
    uint32_t t_out_end = get_arg_val<uint32_t>(2);
    uint32_t h_out_start = get_arg_val<uint32_t>(3);
    uint32_t h_out_end = get_arg_val<uint32_t>(4);
    uint32_t w_out_start = get_arg_val<uint32_t>(5);
    uint32_t w_out_end = get_arg_val<uint32_t>(6);

    // Interleaved address generator for output
    constexpr bool is_dram = true;
    const InterleavedAddrGen<is_dram> out_writer = {.bank_base_address = output_addr, .page_size = out_page_size_bytes};

    // For each output position in our assigned range
    for (uint32_t t_out = t_out_start; t_out < t_out_end; t_out++) {
        for (uint32_t h_out = h_out_start; h_out < h_out_end; h_out++) {
            for (uint32_t w_out = w_out_start; w_out < w_out_end; w_out++) {
                // Wait for compute kernel to produce max pooled result
                cb_wait_front(cb_output, 1);

                // Write output to DRAM - need to get actual output dimensions from compile args
                constexpr uint32_t H_out = get_compile_time_arg_val(2);
                constexpr uint32_t W_out = get_compile_time_arg_val(3);

                uint32_t output_page_idx = (t_out * H_out + h_out) * W_out + w_out;
                uint64_t dst_noc_addr = out_writer.get_noc_addr(output_page_idx);
                noc_async_write(get_read_ptr(cb_output), dst_noc_addr, out_page_size_bytes);
                noc_async_write_barrier();

                cb_pop_front(cb_output, 1);
            }
        }
    }
}
