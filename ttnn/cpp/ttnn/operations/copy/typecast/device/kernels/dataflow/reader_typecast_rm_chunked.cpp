// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t start_row_id = get_arg(args::start_row_id);

    constexpr uint32_t full_chunks_per_row = get_arg(args::full_chunks_per_row);
    constexpr uint32_t partial_chunks_per_row = get_arg(args::partial_chunks_per_row);  // 0 or 1
    constexpr uint32_t full_chunk_size_bytes = get_arg(args::full_chunk_size_bytes);
    constexpr uint32_t partial_chunk_size_bytes = get_arg(args::partial_chunk_size_bytes);

    constexpr uint32_t onepage = 1;

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    const auto s = TensorAccessor(tensor::input);

    const uint32_t end_row_id = start_row_id + num_rows;

    for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
        // Process all full chunks for this row
        for (uint32_t chunk_idx = 0; chunk_idx < full_chunks_per_row; ++chunk_idx) {
            dfb_in.reserve_back(onepage);

            const uint32_t byte_offset = chunk_idx * full_chunk_size_bytes;
#ifdef ARCH_QUASAR
            // On Quasar DM, get_write_ptr() returns the UNCACHED L1 alias and NOC APIs do not
            // accept uncached addresses — pass the DFB endpoint so the NoC uses the cached address.
            noc.async_read(s, dfb_in, full_chunk_size_bytes, {.page_id = row_id, .offset_bytes = byte_offset}, {});
#else
            const uint32_t l1_write_addr = dfb_in.get_write_ptr();
            noc.async_read(
                s,
                CoreLocalMem<uint32_t>(l1_write_addr),
                full_chunk_size_bytes,
                {.page_id = row_id, .offset_bytes = byte_offset},
                {});
#endif

            noc.async_read_barrier();
            dfb_in.push_back(onepage);
        }

        // Process partial chunk if it exists
        if constexpr (partial_chunks_per_row > 0) {
            dfb_in.reserve_back(onepage);

            const uint32_t byte_offset = full_chunks_per_row * full_chunk_size_bytes;
#ifdef ARCH_QUASAR
            // See the full-chunk read above: the NoC needs the DFB's cached address on Quasar.
            noc.async_read(s, dfb_in, partial_chunk_size_bytes, {.page_id = row_id, .offset_bytes = byte_offset}, {});
#else
            const uint32_t l1_write_addr = dfb_in.get_write_ptr();
            noc.async_read(
                s,
                CoreLocalMem<uint32_t>(l1_write_addr),
                partial_chunk_size_bytes,
                {.page_id = row_id, .offset_bytes = byte_offset},
                {});
#endif

            noc.async_read_barrier();
            dfb_in.push_back(onepage);
        }
    }
}
