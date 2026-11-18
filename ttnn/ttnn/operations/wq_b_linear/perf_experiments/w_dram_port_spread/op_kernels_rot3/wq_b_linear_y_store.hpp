// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// The y block (block_m_tiles x core_n_tiles, tile (m, n) at CB slot m * core_n_tiles + n) is stored by two RISCs:
// the reader (NoC0) owns the last reader_tiles slots in column-major (n, m) order, the writer (NoC1) the rest.
FORCE_INLINE uint32_t y_reader_tiles(uint32_t total_tiles, uint32_t share_tenths) {
    return (total_tiles * share_tenths + 5) / 10;
}

FORCE_INLINE bool reader_owns_y_slot(
    uint32_t m, uint32_t n, uint32_t block_m_tiles, uint32_t total, uint32_t reader_tiles) {
    return n * block_m_tiles + m >= total - reader_tiles;
}

// Writes this RISC's share of the y block from cb_y (read pointer = slot 0) to the interleaved y pages
// m * tensor_n_tiles + n_start + n * n_stride. The caller has waited for the block and issues the barrier.
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
    uint32_t reader_tiles,
    bool is_reader) {
    const uint32_t total = block_m_tiles * core_n_tiles;
    uint32_t offset = 0;
    for (uint32_t m = 0; m < block_m_tiles; ++m) {
        const uint32_t row_page = m * tensor_n_tiles + n_start;
        for (uint32_t n = 0; n < core_n_tiles; ++n) {
            if (reader_owns_y_slot(m, n, block_m_tiles, total, reader_tiles) == is_reader) {
                noc.async_write(cb_y, y, y_page_bytes, {.offset_bytes = offset}, {.page_id = row_page + n * n_stride});
            }
            offset += y_page_bytes;
        }
    }
}
