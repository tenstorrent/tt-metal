// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad extension_band: write-only band writer.
// Writes c into rows [band_row_begin, band_row_end) of the last tile row of every slice, in place,
// for this kernel's contiguous range of band tiles (band_idx = slice * in_tiles_w + wt).
// The input is never read; no other tile row is touched.
//
// Block operations (op_design.md, Block schedule / extension_band):
//   E1 make_fill_page  -- fill_l1_range of fill_bits into cb_fill_tile (once per kernel)
//   E2 write_band_block -- per band tile, one NoC write per coalesced band segment, sourced from the
//                          same offset of the fill page (src == dst mod 16 trivially), no barrier inside
//   E3 drain_writes    -- one noc_async_write_barrier per kernel (BB-03)
// No helper covers E2/E3 (pure NoC writes; see design "Helpers considered and rejected").

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(0);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t in_tiles_h = get_compile_time_arg_val(3);
    constexpr uint32_t in_tiles_w = get_compile_time_arg_val(4);
    constexpr uint32_t band_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_segments = get_compile_time_arg_val(6);
    constexpr uint32_t seg_off[4] = {
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(13)};
    constexpr uint32_t seg_len[4] = {
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(14)};
    constexpr auto dst_args = TensorAccessorArgs<15>();
    static_assert(num_segments >= 1 && num_segments <= 4, "band segments: 1..4");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_band = get_arg_val<uint32_t>(1);
    const uint32_t num_band = get_arg_val<uint32_t>(2);
    const uint32_t fill_bits = get_arg_val<uint32_t>(3);

    if (num_band == 0) {
        return;
    }

    const auto dst = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // E1: fill page (scratch owned by this kernel only; never pushed).
    const uint32_t fill_base = get_write_ptr(cb_fill_tile);
    {
        DeviceZoneScopedN("make_fill_page");
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_base, tile_bytes, fill_bits);
    }

    // Walk (slice, wt) incrementally from the flat start index.
    uint32_t wt = start_band % in_tiles_w;
    uint32_t slice = start_band / in_tiles_w;
    constexpr uint32_t last_row_offset = (in_tiles_h - 1) * in_tiles_w;

    const uint32_t num_blocks = (num_band + band_block_tiles - 1) / band_block_tiles;
    uint32_t remaining = num_band;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("write_band_block");
        const uint32_t extent = remaining < band_block_tiles ? remaining : band_block_tiles;
        for (uint32_t k = 0; k < extent; ++k) {
            const uint32_t page = slice * (in_tiles_h * in_tiles_w) + last_row_offset + wt;
            const uint64_t page_noc = dst.get_noc_addr(page);
            for (uint32_t s = 0; s < num_segments; ++s) {
                noc_async_write(fill_base + seg_off[s], page_noc + seg_off[s], seg_len[s]);
            }
            if (++wt == in_tiles_w) {
                wt = 0;
                ++slice;
            }
        }
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
