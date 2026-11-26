// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout writer (BRISC, NoC1) — `store_block` of regime R1.
//
// Per block: one cb_wait_front for the whole block, one write per (live row, unit) of the unit's
// live bytes into ROW_MAJOR page n*H + h at byte offset chunk*block_tile_cols*64, ONE
// noc_async_writes_flushed, one pop. Row r of a unit sits at r * block_tile_cols * 64 B in the CB.
// Rows h >= H of an image's last tile row and columns w >= W are never written as values (the last
// chunk's length is rounded up to the write granule, staying inside the page's aligned slot).
//
// Helper considered and rejected: dataflow_kernel_lib::write_sticks_after_untilize has no width-chunk
// page offset and no per-image live-row count (op_design.md, API Mapping).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(0);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_image_h = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);  // 2W: one logical output row
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t write_granule = get_compile_time_arg_val(8);
    constexpr auto output_args = TensorAccessorArgs<9>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t chunk_bytes = tile_bytes / TILE_H * block_tile_cols;  // block_tile_cols * 64 for bf16
    constexpr uint32_t unit_bytes = block_tile_cols * tile_bytes;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t units_this_core = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr, row_bytes);

    uint32_t tile_row = start_unit / num_col_chunks;
    uint32_t chunk = start_unit % num_col_chunks;

    for (uint32_t done = 0; done < units_this_core; done += block_tile_rows) {
        DeviceZoneScopedN("store_block");
        const uint32_t extent = (units_this_core - done) < block_tile_rows ? (units_this_core - done) : block_tile_rows;
        const uint32_t block_pages = extent * block_tile_cols;
        {
            DeviceZoneScopedN("wait_rows");
            cb_wait_front(cb_output_rows, block_pages);
        }
        uint32_t l1_unit = get_read_ptr(cb_output_rows);

        for (uint32_t u = 0; u < extent; ++u) {
            const uint32_t image = tile_row / tiles_per_image_h;
            const uint32_t h0 = (tile_row - image * tiles_per_image_h) * TILE_H;
            const uint32_t rows_left = H - h0;
            const uint32_t rows_live = rows_left < TILE_H ? rows_left : TILE_H;
            const uint32_t byte_offset = chunk * chunk_bytes;
            const uint32_t bytes_left = row_bytes - byte_offset;
            const uint32_t write_bytes = bytes_left < chunk_bytes
                                             ? ((bytes_left + write_granule - 1) / write_granule) * write_granule
                                             : chunk_bytes;
            uint32_t page = image * H + h0;
            uint32_t l1_row = l1_unit;
            for (uint32_t r = 0; r < rows_live; ++r, ++page, l1_row += chunk_bytes) {
                noc_async_write(l1_row, output_accessor.get_noc_addr(page, byte_offset), write_bytes);
            }
            l1_unit += unit_bytes;
            if (++chunk == num_col_chunks) {
                chunk = 0;
                ++tile_row;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, block_pages);
    }
    noc_async_write_barrier();
}
