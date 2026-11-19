// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

// The tile columns [first_column, first_column + columns) of all k_tiles tile rows, k-major, resident in the CB.
template <typename Accessor>
void load_resident_weights(
    const Noc& noc,
    const Accessor& weights,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t k_tiles,
    uint32_t row_tiles,
    uint32_t first_row,
    uint32_t first_column,
    uint32_t columns) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(k_tiles * columns);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t k = 0; k < k_tiles; k++) {
        for (uint32_t j = 0; j < columns; j++) {
            noc.async_read(
                weights,
                CoreLocalMem<uint32_t>(base + (k * columns + j) * tile_bytes),
                tile_bytes,
                {.page_id = (first_row + k) * row_tiles + first_column + j},
                {});
        }
    }
    noc.async_read_barrier();
    cb.push_back(k_tiles * columns);
}
