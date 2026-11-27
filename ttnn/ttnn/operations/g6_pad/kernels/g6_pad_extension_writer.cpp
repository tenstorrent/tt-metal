// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regime E (tile_padding_extension), BRISC / NoC1, the only kernel of the regime.
//
// Writes c into tile rows [in_row_rem, out_row_end) of the last tile row of every slice of x, in
// place, as <= 3 face-row-aligned segments per tile sourced from cb_fill_tile at the same byte
// offset. Reads nothing; never touches another tile row. cb_fill_tile is self-owned scratch:
// filled through its write pointer once, never pushed or popped.
//
// Raw dataflow for the writes: no kernel_lib helper wraps sub-page interleaved writes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_pad_fill.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_segments = get_compile_time_arg_val(6);
    constexpr uint32_t seg_offset[3] = {
        get_compile_time_arg_val(7), get_compile_time_arg_val(9), get_compile_time_arg_val(11)};
    constexpr uint32_t seg_bytes[3] = {
        get_compile_time_arg_val(8), get_compile_time_arg_val(10), get_compile_time_arg_val(12)};
    constexpr auto x_args = TensorAccessorArgs<13>();
    static_assert(num_segments >= 1 && num_segments <= 3, "E writes 1..3 segments per tile");

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const uint32_t fill_addr = get_write_ptr(cb_fill_tile);

    // build_fill_segments: only the bytes the segments send; zeros as one span (the gaps between
    // segments are never sent, so zeroing them is harmless).
    {
        DeviceZoneScopedN("G6PAD_FILL_BUILD");
        constexpr uint32_t span_begin = seg_offset[0];
        constexpr uint32_t span_end = seg_offset[num_segments - 1] + seg_bytes[num_segments - 1];
        if (fill_bits == 0) {
            g6_pad::fill_constant<elem_bytes>(cb_fill_tile, span_begin, span_end - span_begin, 0);
        } else {
            for (uint32_t k = 0; k < num_segments; ++k) {
                g6_pad::fill_constant<elem_bytes>(cb_fill_tile, seg_offset[k], seg_bytes[k], fill_bits);
            }
        }
    }

    // Work tile w = slice * tensor_col_tiles + tile_col maps to the last tile row of its slice.
    constexpr uint32_t slice_page_stride = tensor_row_tiles * tensor_col_tiles;
    uint32_t tile_col = core_work_begin % tensor_col_tiles;
    const uint32_t slice = core_work_begin / tensor_col_tiles;
    uint32_t page = slice * slice_page_stride + (tensor_row_tiles - 1) * tensor_col_tiles + tile_col;

    uint32_t tiles_left = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        // fill_new_rows_block: no barrier inside the block; the source bytes never change.
        DeviceZoneScopedN("G6PAD_FILL_ROWS");
        const uint32_t block_extent = tiles_left < block_tiles ? tiles_left : block_tiles;
        for (uint32_t i = 0; i < block_extent; ++i) {
            const uint64_t page_noc_addr = x_accessor.get_noc_addr(page);
            for (uint32_t k = 0; k < num_segments; ++k) {
                noc_async_write(fill_addr + seg_offset[k], page_noc_addr + seg_offset[k], seg_bytes[k]);
            }
            ++page;
            if (++tile_col == tensor_col_tiles) {
                tile_col = 0;
                page += slice_page_stride - tensor_col_tiles;
            }
        }
        tiles_left -= block_extent;
    }

    // drain_writes (BB-03): every write acked before the kernel returns.
    {
        DeviceZoneScopedN("G6PAD_DRAIN");
        noc_async_write_barrier();
    }
}
