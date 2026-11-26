// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout writer (RISCV_0 / NoC1) — block operation `store_block`.
//
// cb_rows holds each segment as 32 rows of segment_width * 64 B. For every live row the writer
// issues one write of seg_bytes to output page n*H + ht*32 + i at byte offset
// chunk * segment_width * 64: tile padding rows and columns never reach the output.
// Waits are cumulative per segment so writing starts as soon as compute packs a segment; the
// block's pages are released once, after its writes have left L1.
//
// write_sticks_after_untilize is not used: it maps tile-row r to page r*32 (wrong for multi-image
// h_non_aligned shapes), re-derives the row stride from the row bytes (wrong for a ragged last
// column chunk) and barriers per tile-row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_rows_id = get_compile_time_arg_val(0);
    constexpr uint32_t segment_width = get_compile_time_arg_val(1);
    constexpr uint32_t block_segments = get_compile_time_arg_val(2);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_h_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t height = get_compile_time_arg_val(5);
    constexpr uint32_t width = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(8);
    constexpr auto output_args = TensorAccessorArgs<9>();

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t row_bytes = out_page_bytes;
    constexpr uint32_t row_stride = segment_width * (tile_bytes / TILE_DIM);
    constexpr uint32_t segment_bytes = segment_width * tile_bytes;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_segment = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    DataflowBuffer cb_rows(cb_rows_id);

    const uint32_t end_segment = start_segment + num_segments;
    uint32_t segment = start_segment;
    while (segment < end_segment) {
        DeviceZoneScopedN("store_block");
        const uint32_t remaining = end_segment - segment;
        const uint32_t block_segs = remaining < block_segments ? remaining : block_segments;
        uint32_t block_addr = 0;
        for (uint32_t s = 0; s < block_segs; ++s, ++segment) {
            // Per-segment waits stay unzoned: one zone per segment would exhaust the per-RISC
            // marker budget on narrow tensors (hundreds of segments per core).
            cb_rows.wait_front((s + 1) * segment_width);
            if (s == 0) {
                block_addr = cb_rows.get_read_ptr();
            }
            const uint32_t tile_row = segment / num_col_chunks;
            const uint32_t chunk = segment - tile_row * num_col_chunks;
            const uint32_t n = tile_row / tensor_h_tiles;
            const uint32_t ht = tile_row - n * tensor_h_tiles;
            const uint32_t rows_left = height - ht * TILE_DIM;
            const uint32_t rows_live = rows_left < TILE_DIM ? rows_left : TILE_DIM;
            const uint32_t byte_offset = chunk * row_stride;
            const uint32_t bytes_left = row_bytes - byte_offset;
            const uint32_t seg_bytes = bytes_left < row_stride ? bytes_left : row_stride;
            const uint32_t first_page = n * height + ht * TILE_DIM;

            uint32_t row_addr = block_addr + s * segment_bytes;
            for (uint32_t i = 0; i < rows_live; ++i, row_addr += row_stride) {
                noc_async_write(row_addr, output.get_noc_addr(first_page + i, byte_offset), seg_bytes);
            }
        }
        {
            DeviceZoneScopedN("writes_flushed");
            noc_async_writes_flushed();
        }
        cb_rows.pop_front(block_segs * segment_width);
    }
    {
        DeviceZoneScopedN("write_barrier");
        noc_async_write_barrier();
    }
}
