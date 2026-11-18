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

// Early release (compute pushes the y block in pieces on its last K-block):
//   kColumn: column n is ready once wait_front(n + 1) passes; tiles of a column sit at m * core_n_tiles + n.
//   kRow: row m is ready once wait_front((m + 1) * core_n_tiles) passes.
// split_front = false: within each released group the reader owns the last round(group size * share / 10) tiles.
// split_front = true: the reader owns the first round(groups * share / 10) whole groups (released earliest).
constexpr uint32_t kYReleaseNone = 0;
constexpr uint32_t kYReleaseColumn = 1;
constexpr uint32_t kYReleaseRow = 2;

template <uint32_t mode, bool split_front, typename Accessor>
FORCE_INLINE void store_y_share_released(
    Noc& noc,
    CircularBuffer& cb_y,
    const Accessor& y,
    uint32_t y_page_bytes,
    uint32_t block_m_tiles,
    uint32_t core_n_tiles,
    uint32_t tensor_n_tiles,
    uint32_t n_start,
    uint32_t n_stride,
    uint32_t share_tenths,
    bool is_reader) {
    constexpr bool by_column = mode == kYReleaseColumn;
    const uint32_t num_groups = by_column ? core_n_tiles : block_m_tiles;
    const uint32_t group_size = by_column ? block_m_tiles : core_n_tiles;
    uint32_t first = 0, end = group_size, g_first = 0, g_end = num_groups;
    if constexpr (split_front) {
        const uint32_t reader_groups = y_reader_tiles(num_groups, share_tenths);
        (is_reader ? g_end : g_first) = reader_groups;
    } else {
        const uint32_t reader_count = y_reader_tiles(group_size, share_tenths);
        (is_reader ? first : end) = group_size - reader_count;
    }
    if (first == end || g_first == g_end) {
        return;
    }
    for (uint32_t g = g_first; g < g_end; ++g) {
        cb_y.wait_front(by_column ? g + 1 : (g + 1) * core_n_tiles);
        for (uint32_t e = first; e < end; ++e) {
            const uint32_t m = by_column ? e : g;
            const uint32_t n = by_column ? g : e;
            noc.async_write(
                cb_y,
                y,
                y_page_bytes,
                {.offset_bytes = (m * core_n_tiles + n) * y_page_bytes},
                {.page_id = m * tensor_n_tiles + n_start + n * n_stride});
        }
    }
}

FORCE_INLINE uint32_t y_released_reader_tiles(
    uint32_t mode, bool split_front, uint32_t block_m_tiles, uint32_t core_n_tiles, uint32_t share_tenths) {
    if (split_front) {
        return y_reader_tiles(mode == kYReleaseColumn ? core_n_tiles : block_m_tiles, share_tenths);
    }
    if (mode == kYReleaseColumn) {
        return core_n_tiles * y_reader_tiles(block_m_tiles, share_tenths);
    }
    return block_m_tiles * y_reader_tiles(core_n_tiles, share_tenths);
}
