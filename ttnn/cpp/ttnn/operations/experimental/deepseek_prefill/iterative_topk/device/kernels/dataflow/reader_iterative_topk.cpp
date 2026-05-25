// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);

    constexpr uint32_t accessor_offset = 2;
    constexpr auto input_args = TensorAccessorArgs<accessor_offset>();
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    for (uint32_t row = start_row; row < end_row; row++) {
        cb_reserve_back(cb_input, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_input);
        noc_async_read_page(row, input_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }
}
