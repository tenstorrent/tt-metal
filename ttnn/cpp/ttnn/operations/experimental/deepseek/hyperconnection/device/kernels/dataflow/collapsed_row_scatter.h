// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// Writes tiles [first_tile, first_tile + num_tiles) of a core's collapsed output shard: the row-major
// collapse leaves the collapsed row in cb_row (as h x 32 tiles, i.e. the row itself), and each
// 32-column run of it becomes row 0 of a zeroed 32x32 output tile. Both data-movement RISCs call
// this on their own halves of the shard, each on its own NoC.
inline void scatter_collapsed_row(
    uint32_t cb_row,
    uint32_t row_chunks,
    uint32_t cb_tiles,
    uint32_t tiles_addr,
    uint32_t first_tile,
    uint32_t num_tiles) {
    constexpr uint32_t tile_bytes = 32 * 32 * sizeof(uint16_t);
    constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
    constexpr uint32_t face_bytes = 16 * face_row_bytes;
    Noc noc;
    CircularBuffer cb_out(cb_tiles);
    noc.async_write_zeros(cb_out, num_tiles * tile_bytes, {.offset_bytes = first_tile * tile_bytes});
    noc.write_zeros_l1_barrier();

    // Both RISCs read the row; neither pops it, the program ends with it.
    CircularBuffer cb_y(cb_row);
    cb_y.wait_front(row_chunks);
    const uint32_t row = cb_y.get_read_ptr();
    noc_async_write_one_packet_set_state(get_noc_addr(tiles_addr), face_row_bytes, noc_index);
    for (uint32_t n = first_tile; n < first_tile + num_tiles; ++n) {
        const uint32_t src = row + n * 2 * face_row_bytes;
        const uint32_t dst = tiles_addr + n * tile_bytes;
        noc_async_write_one_packet_with_state(src, dst, noc_index);
        noc_async_write_one_packet_with_state(src + face_row_bytes, dst + face_bytes, noc_index);
    }
    noc_async_write_barrier(noc_index);
}
