// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// read_coalescing example: reader. See README.md for the four-arm experiment.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr bool bank_affinity = get_compile_time_arg_val(4) != 0;
    constexpr bool coalesce_reads = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t pages_per_row = get_compile_time_arg_val(6);
    constexpr auto in_args = TensorAccessorArgs<7>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);  // shards per row
    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

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
            const uint32_t page_in_row = col_bytes / page_bytes;  // the shard, i.e. the bank
            const uint32_t byte_offset = col_bytes % page_bytes;

            cb_reserve_back(cb_in, 1);
            const uint32_t l1_addr = get_write_ptr(cb_in);
            // A full 32-row block is contiguous within this bank and the reserved CB slot.
            // One wide API call replaces 32 stick reads. noc_async_read handles hardware
            // burst splitting when the block exceeds NOC_MAX_BURST_SIZE.
            constexpr uint32_t rows_per_read = coalesce_reads ? block_rows : 1;
            for (uint32_t r = 0; r < block_rows; r += rows_per_read) {
                noc_async_read(
                    in_acc.get_noc_addr((row0 + r) * pages_per_row + page_in_row, byte_offset),
                    l1_addr + r * chunk_row_bytes,
                    rows_per_read * chunk_row_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
