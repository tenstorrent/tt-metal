// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize 2D-work-split writer (BRISC / NoC1) — interleaved width-split path.
//
// Drains the Wt_chunk TILE pages produced per work unit from cb_tiled_out and
// writes them to their interleaved output pages. Mirrors tilize_writer.cpp's
// batched raw noc_async_write (one barrier per block) — there is no tile-page
// writer helper — but keys off the flat unit index rather than a contiguous
// tile-row range.
//
// A unit u = (row, chunk) with row = u / C, chunk = u % C (C = Wt / Wt_chunk).
// Its Wt_chunk output tiles occupy pages
//   base_page = row*Wt + chunk*Wt_chunk, base_page+1, ... base_page+Wt_chunk-1
// in the tile-row-major output page order page(tr, tc) = tr*Wt + tc. Reader push
// order == this pop order == compute FIFO order, so the k-th unit's tiles are the
// k-th block in the CB.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiled_out = 16;
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);              // full width in tiles
    constexpr uint32_t Wt_chunk = get_compile_time_arg_val(2);        // tiles per unit
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(3);  // C = Wt / Wt_chunk
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_start = get_arg_val<uint32_t>(1);  // per-core first unit
    const uint32_t u_count = get_arg_val<uint32_t>(2);  // per-core unit count

    const auto accessor = TensorAccessor(dst_args, dst_addr, out_tile_size);

    for (uint32_t u = u_start; u < u_start + u_count; ++u) {
        const uint32_t row = u / chunks_per_row;
        const uint32_t chunk = u - row * chunks_per_row;
        cb_wait_front(cb_tiled_out, Wt_chunk);
        const uint32_t l1_addr = get_read_ptr(cb_tiled_out);
        const uint32_t base_page = row * Wt + chunk * Wt_chunk;
        for (uint32_t i = 0; i < Wt_chunk; ++i) {
            noc_async_write(l1_addr + i * out_tile_size, accessor.get_noc_addr(base_page + i), out_tile_size);
        }
        noc_async_write_barrier();  // ONE barrier for Wt_chunk writes
        cb_pop_front(cb_tiled_out, Wt_chunk);
    }
}
