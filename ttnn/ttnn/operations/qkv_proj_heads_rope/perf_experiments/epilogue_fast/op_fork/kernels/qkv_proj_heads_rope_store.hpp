// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// store_row: one output token row (n_tiles bf16 tiles at the front of cb_out) -> q|k|v head layout.
// Shared by the writer (NoC1) and the reader (NoC0): output rows alternate between the two kernels'
// output CBs so the store tail uses both NoCs.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

namespace qkv_store {

template <
    uint32_t cb_out,
    uint32_t n_tiles,
    uint32_t tiles_per_head,
    uint32_t num_heads,
    uint32_t seq_tiles,
    uint32_t tile_bytes,
    uint32_t last_valid_rows,
    typename QAccessor,
    typename KAccessor,
    typename VAccessor>
FORCE_INLINE void store_row(
    Noc& noc,
    uint32_t r,  // global tile-row index (b * seq_tiles + st)
    uint32_t third,
    uint32_t first_head,
    const QAccessor& q_accessor,
    const KAccessor& k_accessor,
    const VAccessor& v_accessor) {
    constexpr uint32_t face_row_bytes = 16 * 2;
    constexpr uint32_t face_bytes = 16 * face_row_bytes;
    const uint32_t batch = r / seq_tiles;
    const uint32_t st = r % seq_tiles;
    cb_wait_front(cb_out, n_tiles);
    uint32_t l1_addr = get_read_ptr(cb_out);
    if constexpr (last_valid_rows != 0) {
        // Output pad rows must be exactly 0 whatever x / table padding held.
        if (st == seq_tiles - 1) {
            CircularBuffer out_tiles(cb_out);
            constexpr uint32_t top_face = last_valid_rows / 16;
            constexpr uint32_t first_row = last_valid_rows % 16;
            constexpr uint32_t partial_bytes = (16 - first_row) * face_row_bytes;
            constexpr uint32_t full_off = (top_face + 1) * 2 * face_bytes;
            // async_write_zeros addresses a CB from its write pointer; the tiles sit at the read
            // pointer (unsigned wrap keeps the delta exact).
            const uint32_t read_off = l1_addr - out_tiles.get_write_ptr();
            for (uint32_t n = 0; n < n_tiles; ++n) {
                const uint32_t tile_off = read_off + n * tile_bytes;
                const uint32_t face_off = tile_off + top_face * 2 * face_bytes + first_row * face_row_bytes;
                noc.async_write_zeros(out_tiles, partial_bytes, {.offset_bytes = face_off});
                noc.async_write_zeros(out_tiles, partial_bytes, {.offset_bytes = face_off + face_bytes});
                if constexpr (full_off < 4 * face_bytes) {
                    noc.async_write_zeros(out_tiles, 4 * face_bytes - full_off, {.offset_bytes = tile_off + full_off});
                }
            }
            noc.write_zeros_l1_barrier();
        }
    }
    for (uint32_t n = 0; n < n_tiles; ++n) {
        const uint32_t head = first_head + n / tiles_per_head;
        const uint32_t page = ((batch * num_heads + head) * seq_tiles + st) * tiles_per_head + n % tiles_per_head;
#ifdef ABL_NO_STORE  // fork-only ablation: stores stubbed (output garbage)
        continue;
#endif
        if (third == 0) {
            noc_async_write_page(page, q_accessor, l1_addr);
        } else if (third == 1) {
            noc_async_write_page(page, k_accessor, l1_addr);
        } else {
            noc_async_write_page(page, v_accessor, l1_addr);
        }
        l1_addr += tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_out, n_tiles);
}

}  // namespace qkv_store
