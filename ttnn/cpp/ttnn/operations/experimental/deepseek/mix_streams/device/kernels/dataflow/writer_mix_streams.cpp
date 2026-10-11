// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Streams this core's slice of the mixed output back, one 32x32 tile at a time: the whole tile
// page for a TILE output, or its rows 0..hc-1 into the token's row pages for a ROW_MAJOR one.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr bool out_is_rm = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t hc = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t out_elems_per_page = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_w = 32;
    // Bytes of one 16-column face row, and the offset of face 1 (columns 16..31) in a tile.
    constexpr uint32_t half_row_bytes = 16 * sizeof(uint16_t);
    constexpr uint32_t face1_bytes = 256 * sizeof(uint16_t);
    constexpr uint32_t out_pages_per_row = (n_tiles * tile_w) / out_elems_per_page;
    const uint32_t tile_size_bytes = out_cb.get_tile_size();

    for (uint32_t page = start_tile; page < start_tile + num_tiles; ++page) {
        out_cb.wait_front(one_tile);
        if constexpr (out_is_rm) {
            const uint32_t t = page / n_tiles;
            const uint32_t col = (page % n_tiles) * tile_w;
            const uint32_t col_page = col / out_elems_per_page;
            const uint32_t col_offset = (col % out_elems_per_page) * sizeof(uint16_t);
            for (uint32_t r = 0; r < hc; ++r) {
                const uint32_t page_id = (t * hc + r) * out_pages_per_row + col_page;
                noc.async_write(
                    out_cb,
                    out,
                    half_row_bytes,
                    {.offset_bytes = r * half_row_bytes},
                    {.page_id = page_id, .offset_bytes = col_offset});
                noc.async_write(
                    out_cb,
                    out,
                    half_row_bytes,
                    {.offset_bytes = face1_bytes + r * half_row_bytes},
                    {.page_id = page_id, .offset_bytes = col_offset + half_row_bytes});
            }
        } else {
            noc.async_write(out_cb, out, tile_size_bytes, {.offset_bytes = 0}, {.page_id = page});
        }
        noc.async_write_barrier();
        out_cb.pop_front(one_tile);
    }
}
