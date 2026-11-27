// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC / NoC1) — block operation `store_block` (op_design.md, Block schedule).
// Drains cb_output_tiles in blocks of `block_tiles` (a ragged last block waits/pops its actual
// extent, matching reader and compute), writes the tiles of each block to [start_tile, start_tile + core_tiles),
// flushes before releasing the slot, and ends with one write barrier so no transaction is in flight at exit (BB-03).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_typecast_rm_rows.hpp"

// rm_mode (regime R2 `rm_pseudo_tile_stream`): mirror of the reader. Per block of `rows_per_block` rows:
// wait the block's ceil(rows * W / 1024) dense pseudo-tile pages, expand the rows into bank-run order in
// the 64B-aligned staging zone (aligned-page stride), release the CB slot, then ONE NoC write per bank run
// (rows r, r+nb, ... are adjacent in their bank). The staging zone is reused only after the previous
// block's writes are flushed. Padding bytes of each aligned page slot are don't-care.

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rm_mode = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_staging = get_compile_time_arg_val(4);
    constexpr uint32_t rows_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t values_per_row = get_compile_time_arg_val(7);
    constexpr uint32_t staging_align = get_compile_time_arg_val(8);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(9);
    constexpr auto output_args = TensorAccessorArgs<10>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    // out_page_bytes is the aligned page size (TILE: the tile; RM: the aligned row slot).
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_page_bytes);

    if constexpr (rm_mode) {
        // start_tile / core_tiles carry start_row / core_rows here.
        // staging_depth >= 2: block b is expanded into its own staging slot while block b-1's writes are
        // still leaving the other slot; the flush after the expansion retires b-1 (the only block in flight),
        // which also frees slot (b+1) % depth before the next expansion.
        constexpr uint32_t pseudo_tile_values = 1024;
        const uint32_t nb = output_accessor.contiguous_page_stride();
        const uint32_t staging0 = (get_write_ptr(cb_output_staging) + staging_align - 1) & ~(staging_align - 1);
        const uint32_t staging_bytes = rows_per_block * out_page_bytes;
        const uint32_t end_row = start_tile + core_tiles;
        uint32_t stage = 0;
        for (uint32_t blk = start_tile; blk < end_row; blk += rows_per_block) {
            DeviceZoneScopedN("WRITE_BLOCK");
            const uint32_t rows = (end_row - blk < rows_per_block) ? (end_row - blk) : rows_per_block;
            const uint32_t block_pages = (rows * values_per_row + pseudo_tile_values - 1) / pseudo_tile_values;
            const uint32_t runs = rows < nb ? rows : nb;
            const uint32_t staging = staging0 + stage * staging_bytes;
            {
                DeviceZoneScopedN("WRITE_WAIT_DATA");
                cb_wait_front(cb_output_tiles, block_pages);
            }
            if constexpr (staging_depth == 1) {
                noc_async_writes_flushed();  // previous block's writes have left the (single) staging zone
            }
            g6_rm::expand_block<row_bytes>(get_read_ptr(cb_output_tiles), staging, out_page_bytes, nb, rows);
            cb_pop_front(cb_output_tiles, block_pages);
            if constexpr (staging_depth > 1) {
                noc_async_writes_flushed();  // block blk-1 left its slot (overlapped with the expansion above)
            }
            uint32_t slot = 0;
            for (uint32_t j = 0; j < runs; ++j) {
                const uint32_t cnt = (rows - j - 1) / nb + 1;
                noc_async_write(
                    staging + slot * out_page_bytes, output_accessor.get_noc_addr(blk + j), cnt * out_page_bytes);
                slot += cnt;
            }
            stage = (stage + 1 == staging_depth) ? 0 : stage + 1;
        }
        noc_async_write_barrier();
        return;
    }

    const uint32_t end_tile = start_tile + core_tiles;
    for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
        DeviceZoneScopedN("WRITE_BLOCK");
        const uint32_t block_extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
        {
            DeviceZoneScopedN("WRITE_WAIT_DATA");
            cb_wait_front(cb_output_tiles, block_extent);
        }
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);
        for (uint32_t t = 0; t < block_extent; ++t) {
            noc_async_write_page(block_start + t, output_accessor, l1_addr);
            l1_addr += out_page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_extent);
    }
    noc_async_write_barrier();
}
