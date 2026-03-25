// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t matrix_page_size = get_compile_time_arg_val(0);  // C * sizeof(float)
    constexpr uint32_t matrix_size = get_compile_time_arg_val(1);       // C (square dim)
    constexpr auto input_args = TensorAccessorArgs<2>();

    // Runtime args
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_batch = get_arg_val<uint32_t>(1);
    uint32_t end_batch = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t onepage = 1;

    const auto src = TensorAccessor(input_args, input_buffer_address, matrix_page_size);

    // For each batch element, read C rows into the CB
    for (uint32_t b = start_batch; b < end_batch; b++) {
        uint32_t base_row = b * matrix_size;
        for (uint32_t row = 0; row < matrix_size; row++) {
            cb_reserve_back(cb_in, onepage);
            uint32_t l1_addr = get_write_ptr(cb_in);
            uint64_t noc_addr = src.get_noc_addr(base_row + row);
            noc_async_read(noc_addr, l1_addr, matrix_page_size);
            noc_async_read_barrier();
            cb_push_back(cb_in, onepage);
        }
    }
}
