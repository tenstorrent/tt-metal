// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// permute reader (NoC0) — `load_block` of the whole_tile_relocation regime.
//
// Each core owns a contiguous range of the linear OUTPUT tile index. A block is
// up to BLOCK_TILES consecutive output tiles that lie inside one (ht,wt) plane;
// because `dims` preserves the inner pair, those map to an equally contiguous
// run of INPUT tile pages, so one base index + increment addresses the whole
// run. BLOCK_TILES whole-page reads are issued before a SINGLE barrier.
//
// The CB always advances in whole BLOCK_TILES units (a short, ragged block
// still reserves/pushes BLOCK_TILES pages) so a block's pages are never split
// by the CB wrap; only `run` of them carry data.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiles = 0;

    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);  // BLOCK_TILES knob
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_plane = get_compile_time_arg_val(2);  // contiguity clamp
    constexpr uint32_t rank = get_compile_time_arg_val(3);
    constexpr uint32_t out_dim[4] = {
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)};
    // coeff[d] = input tile stride of the input axis that output axis d came from
    constexpr uint32_t coeff[4] = {
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11)};
    constexpr auto in_args = TensorAccessorArgs<12>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t tiles_this_core = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    uint32_t out_idx = start_tile;
    uint32_t left = tiles_this_core;

    while (left > 0) {
        // --- block extent: BLOCK_TILES, clamped by plane and by core range ---
        const uint32_t off_in_plane = out_idx % tiles_per_plane;
        uint32_t run = block_tiles;
        const uint32_t plane_left = tiles_per_plane - off_in_plane;
        if (plane_left < run) {
            run = plane_left;
        }
        if (left < run) {
            run = left;
        }

        // --- output tile index -> input tile index (permutation as strides) ---
        uint32_t rem = out_idx;
        uint32_t in_base = 0;
        for (uint32_t d = rank; d-- > 0;) {
            const uint32_t c = rem % out_dim[d];
            rem /= out_dim[d];
            in_base += c * coeff[d];
        }

        cb_reserve_back(cb_tiles, block_tiles);
        const uint32_t l1_write_addr = get_write_ptr(cb_tiles);
        for (uint32_t i = 0; i < run; ++i) {
            noc_async_read(in_acc.get_noc_addr(in_base + i), l1_write_addr + i * page_bytes, page_bytes);
        }
        noc_async_read_barrier();  // one barrier per block
        cb_push_back(cb_tiles, block_tiles);

        out_idx += run;
        left -= run;
    }
}
