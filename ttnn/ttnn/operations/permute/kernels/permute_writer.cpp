// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// permute writer (NoC1) — `store_block` of the whole_tile_relocation regime.
//
// Mirror image of the reader: the same block extent rule (BLOCK_TILES clamped
// by plane remainder and core remainder) so the CB quanta match exactly, and
// BLOCK_TILES whole-page writes per SINGLE barrier. Output pages of a block are
// consecutive by construction (the core owns a linear output-tile range).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiles = 0;

    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_plane = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<12>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t tiles_this_core = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);

    uint32_t out_idx = start_tile;
    uint32_t left = tiles_this_core;

    while (left > 0) {
        const uint32_t off_in_plane = out_idx % tiles_per_plane;
        uint32_t run = block_tiles;
        const uint32_t plane_left = tiles_per_plane - off_in_plane;
        if (plane_left < run) {
            run = plane_left;
        }
        if (left < run) {
            run = left;
        }

        cb_wait_front(cb_tiles, block_tiles);
        const uint32_t l1_read_addr = get_read_ptr(cb_tiles);
        for (uint32_t i = 0; i < run; ++i) {
            noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(out_idx + i), page_bytes);
        }
        noc_async_write_barrier();  // one barrier per block
        cb_pop_front(cb_tiles, block_tiles);

        out_idx += run;
        left -= run;
    }
}
