// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t rm_page_size = get_compile_time_arg_val(2);
    constexpr auto output_ta_args = TensorAccessorArgs<3>();

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rm_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    // Tensor accessor uses RM page size (output tensor's actual page size)
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, rm_page_size);

    uint32_t page_id = start_page_id;
    uint32_t pages_remaining = num_rm_pages;

    // Each untilize block produces Wt tile-pages containing 32 contiguous RM rows
    while (pages_remaining > 0) {
        uint32_t rows_this_block = (pages_remaining >= 32) ? 32 : pages_remaining;

        // Wait for Wt tile-pages from untilize (data is contiguous row-major)
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write each RM row individually to the output tensor
        for (uint32_t row = 0; row < rows_this_block; row++) {
            noc_async_write_page(page_id, output_accessor, l1_read_addr);
            l1_read_addr += rm_page_size;
            page_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
        pages_remaining -= rows_this_block;
    }
}
