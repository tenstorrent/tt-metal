// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (op-private kernel; used only by the migrated TypecastRowMajorChunkedProgramFactory).
// Only the binding mechanism changed: the destination address comes from the TensorAccessor binding
// (ta::), the CB id and structural chunk scalars from the DFB/compile-time tokens (dfb:: / args::), and
// the per-core row range from named runtime args (args::). The chunked write loop is preserved.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t start_row_id = get_arg(args::start_row_id);

    constexpr uint32_t cb_id_out = dfb::output_cb;
    constexpr uint32_t full_chunks_per_row = get_arg(args::full_chunks_per_row);
    constexpr uint32_t partial_chunks_per_row = get_arg(args::partial_chunks_per_row);  // 0 or 1
    constexpr uint32_t full_chunk_size_bytes = get_arg(args::full_chunk_size_bytes);
    constexpr uint32_t partial_chunk_size_bytes = get_arg(args::partial_chunk_size_bytes);

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::dst_args);

    const uint32_t end_row_id = start_row_id + num_rows;

    for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
        // Process all full chunks for this row
        for (uint32_t chunk_idx = 0; chunk_idx < full_chunks_per_row; ++chunk_idx) {
            cb_wait_front(cb_id_out, onepage);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_out);

            const uint32_t byte_offset = chunk_idx * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_write(l1_read_addr, chunk_noc_addr, full_chunk_size_bytes);

            noc_async_writes_flushed();
            cb_pop_front(cb_id_out, onepage);
        }

        // Process partial chunk if it exists
        if constexpr (partial_chunks_per_row > 0) {
            cb_wait_front(cb_id_out, onepage);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_out);

            const uint32_t byte_offset = full_chunks_per_row * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_write(l1_read_addr, chunk_noc_addr, partial_chunk_size_bytes);

            noc_async_writes_flushed();
            cb_pop_front(cb_id_out, onepage);
        }
    }
    noc_async_write_barrier();
}
