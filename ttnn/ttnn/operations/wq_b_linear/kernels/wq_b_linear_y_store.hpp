// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// The y block (block_m_tiles x core_n_tiles, tile (m, n) at CB slot m * core_n_tiles + n) is stored by two RISCs,
// column by column as compute releases them: column n is ready once wait_front(n + 1) passes. In every column the
// reader (NoC0) owns the last y_reader_tiles(block_m_tiles, share) rows, the writer (NoC1) the rest.
FORCE_INLINE uint32_t y_reader_tiles(uint32_t total_tiles, uint32_t share_tenths) {
    return (total_tiles * share_tenths + 5) / 10;
}

// Waits for each column, then writes this RISC's rows of it from cb_y (read pointer = slot 0) to the interleaved
// y pages m * tensor_n_tiles + n_start + n * n_stride. The caller issues the barrier.
template <typename Accessor>
FORCE_INLINE void store_y_share(
    Noc& noc,
    CircularBuffer& cb_y,
    const Accessor& y,
    uint32_t y_page_bytes,
    uint32_t block_m_tiles,
    uint32_t core_n_tiles,
    uint32_t tensor_n_tiles,
    uint32_t n_start,
    uint32_t n_stride,
    uint32_t reader_rows,
    bool is_reader) {
    const uint32_t m_first = is_reader ? block_m_tiles - reader_rows : 0;
    const uint32_t m_end = is_reader ? block_m_tiles : block_m_tiles - reader_rows;
    if (m_first == m_end) {
        return;
    }
    for (uint32_t n = 0; n < core_n_tiles; ++n) {
        cb_y.wait_front(n + 1);
        for (uint32_t m = m_first; m < m_end; ++m) {
            noc.async_write(
                cb_y,
                y,
                y_page_bytes,
                {.offset_bytes = (m * core_n_tiles + n) * y_page_bytes},
                {.page_id = m * tensor_n_tiles + n_start + n * n_stride});
        }
    }
}
