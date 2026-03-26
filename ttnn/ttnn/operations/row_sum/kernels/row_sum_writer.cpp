// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Sum - Writer Kernel
// Writes output to DRAM. Supports tiled (tile streaming) and row-major
// (write_sticks_after_untilize) outputs.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    // Compile-time args
    constexpr bool output_is_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t arg2 = get_compile_time_arg_val(2);  // num_output_tiles (tiled) or H (RM)
    constexpr uint32_t arg3 = get_compile_time_arg_val(3);  // 0 (tiled) or output_row_bytes (RM)

    // TensorAccessor compile-time args start at index 4
    constexpr auto accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);

    if constexpr (output_is_rm) {
        // Row-major: use write_sticks_after_untilize helper
        constexpr uint32_t H = arg2;
        constexpr uint32_t row_bytes = arg3;
        const auto accessor = TensorAccessor(accessor_args, output_addr);
        dataflow_kernel_lib::write_sticks_after_untilize<cb_out>(accessor, H, row_bytes, start_page);
    } else {
        // Tiled: stream tiles one at a time
        constexpr uint32_t num_tiles = arg2;
        const auto accessor = TensorAccessor(accessor_args, output_addr);

        for (uint32_t i = start_page; i < start_page + num_tiles; i++) {
            cb_wait_front(cb_out, 1);
            uint32_t l1_addr = get_read_ptr(cb_out);
            noc_async_write_tile(i, accessor, l1_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
