// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1).
//
// Drains TILE pages from cb_tiled_out and writes them to their interleaved
// output tensor pages. Runs on NoC1 so its writes overlap the reader's NoC0
// reads. There is no tile-page writer helper in the dataflow kernel lib
// (write_sticks_after_untilize emits row-major sticks and would destroy the
// tile layout), so this is the canonical batched raw noc_async_write loop:
// one barrier per block (Wt_chunk writes in flight).
//
// Output tile page order is row-major over tiles: page(tr, tc) = tr * Wt + tc,
// with all leading dims folded into the tile-row index tr.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiled_out = 16;
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);        // full width in tiles
    constexpr uint32_t Wt_chunk = get_compile_time_arg_val(2);  // tiles per block
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);  // per-core tile-row offset
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);      // per-core tile-row count

    const auto accessor = TensorAccessor(dst_args, dst_addr, out_tile_size);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(cb_tiled_out, Wt_chunk);
            const uint32_t l1_addr = get_read_ptr(cb_tiled_out);
            const uint32_t base_page = (start_tile_row + block) * Wt + chunk * Wt_chunk;
            for (uint32_t i = 0; i < Wt_chunk; ++i) {
                noc_async_write(l1_addr + i * out_tile_size, accessor.get_noc_addr(base_page + i), out_tile_size);
            }
            noc_async_write_barrier();  // ONE barrier for Wt_chunk writes
            cb_pop_front(cb_tiled_out, Wt_chunk);
        }
    }
}
