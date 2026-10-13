// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// read_coalescing example: writer. See README.md for the four-arm experiment.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_data = 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(0);
    constexpr uint32_t output_row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr bool bank_affinity = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);  // shards per row
    const auto out_acc = TensorAccessor(out_args, dst_addr, output_row_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t j = 0; j < unit_count; ++j) {
            uint32_t u;
            if constexpr (bank_affinity) {
                u = unit_start + j * n_w;  // fixed bank, successive row blocks
            } else {
                u = unit_start + j;  // row-major walk across banks
            }
            const uint32_t row0 = (u / n_w) * block_rows;
            const uint32_t col_bytes = (u % n_w) * chunk_row_bytes;

            cb_wait_front(cb_data, 1);
            const uint32_t l1_addr = get_read_ptr(cb_data);
            // Same output transfers in every arm: preserve the original row-major layout.
            for (uint32_t r = 0; r < block_rows; ++r) {
                noc_async_write(
                    l1_addr + r * chunk_row_bytes, out_acc.get_noc_addr(row0 + r, col_bytes), chunk_row_bytes);
            }
            // The CB slot can be recycled once the data has left L1; no need to wait for acks.
            noc_async_writes_flushed();
            cb_pop_front(cb_data, 1);
        }
    }
    noc_async_write_barrier();
}
