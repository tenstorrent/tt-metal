// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t full_chunk_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t full_chunks_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t partial_chunk_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t partial_chunks_per_row = get_compile_time_arg_val(4);  // 0 or 1
    constexpr uint32_t row_page_size_bytes = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t onepage = 1;

    // Create TensorAccessor with row page size (buffer's actual layout)
    const auto s = TensorAccessor(src_args, src_addr, row_page_size_bytes);

    const uint32_t end_row_id = start_row_id + num_rows;

    for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
        // Process all full chunks for this row
        for (uint32_t chunk_idx = 0; chunk_idx < full_chunks_per_row; ++chunk_idx) {
            cb_reserve_back(cb_id_in, onepage);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            const uint32_t byte_offset = chunk_idx * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_read(chunk_noc_addr, l1_write_addr, full_chunk_size_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_id_in, onepage);
        }

        // Process partial chunk if it exists
        if constexpr (partial_chunks_per_row > 0) {
            cb_reserve_back(cb_id_in, onepage);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            const uint32_t byte_offset = full_chunks_per_row * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_read(chunk_noc_addr, l1_write_addr, partial_chunk_size_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_id_in, onepage);
        }
    }
}
