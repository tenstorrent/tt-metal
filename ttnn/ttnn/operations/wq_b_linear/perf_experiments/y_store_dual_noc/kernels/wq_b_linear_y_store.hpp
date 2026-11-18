// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// The y block (block_m_tiles x core_n_tiles, tile (m, n) at CB slot m * core_n_tiles + n) is stored by two RISCs:
// the reader (NoC0) owns reader_tiles of the block's T tiles, the writer (NoC1) the rest.
//   order 0: the reader owns the last reader_tiles slots in row-major (m, n) order.
//   order 1: the reader owns the last reader_tiles slots in column-major (n, m) order (whole columns first).
//   order 2: the reader owns every other slot (odd row-major index), up to reader_tiles of them.
enum class YSplitOrder : uint32_t { RowMajor = 0, ColMajor = 1, Alternate = 2 };

FORCE_INLINE uint32_t y_reader_tiles(uint32_t total_tiles, uint32_t share_tenths) {
    return (total_tiles * share_tenths + 5) / 10;
}

FORCE_INLINE bool reader_owns_y_slot(
    YSplitOrder order, uint32_t m, uint32_t n, uint32_t block_m_tiles, uint32_t core_n_tiles, uint32_t reader_tiles) {
    const uint32_t total = block_m_tiles * core_n_tiles;
    if (order == YSplitOrder::RowMajor) {
        return m * core_n_tiles + n >= total - reader_tiles;
    }
    if (order == YSplitOrder::ColMajor) {
        return n * block_m_tiles + m >= total - reader_tiles;
    }
    const uint32_t f = m * core_n_tiles + n;
    return (f & 1) && (f >> 1) < reader_tiles;
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
    YSplitOrder order,
    uint32_t reader_tiles,
    bool is_reader) {
    uint32_t offset = 0;
    for (uint32_t m = 0; m < block_m_tiles; ++m) {
        const uint32_t row_page = m * tensor_n_tiles + n_start;
        for (uint32_t n = 0; n < core_n_tiles; ++n) {
            if (reader_owns_y_slot(order, m, n, block_m_tiles, core_n_tiles, reader_tiles) == is_reader) {
                noc.async_write(cb_y, y, y_page_bytes, {.offset_bytes = offset}, {.page_id = row_page + n * n_stride});
            }
            offset += y_page_bytes;
        }
    }
}
