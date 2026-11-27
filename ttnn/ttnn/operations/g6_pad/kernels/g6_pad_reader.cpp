// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage reader (RISCV_1 / NOC0), tiles and rows (CT flag rows_mode).
// rows: unit = (output row, row chunk); source-bearing chunks read the input part of the chunk only.
// N2 load_block: reserve block_tiles slots; for every source-bearing output tile k of the block, read
// its single input tile into slot k; ONE read barrier; push block_tiles (nominal -- fill-only slots and
// a ragged last block are simply left unwritten, so every ring cycle closes at capacity).
// Fill-only output tiles are never read (the writer sources them from cb_fill_tile).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_pad_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(1);  // CB page: one tile, or one row-chunk slot
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_tiles_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_tiles_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(5);
    constexpr uint32_t in_tiles_h = get_compile_time_arg_val(6);
    constexpr uint32_t in_tiles_w = get_compile_time_arg_val(7);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(8);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(9);
    constexpr bool rows_mode = get_compile_time_arg_val(10) != 0;      // new_storage_rows (ROW_MAJOR)
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(11);     // rows: bytes per row chunk
    constexpr uint32_t in_row_bytes = get_compile_time_arg_val(12);    // rows: logical input row bytes
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(13);  // aligned input page size
    constexpr auto src_args = TensorAccessorArgs<14>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    const auto src = TensorAccessor(src_args, src_addr, src_page_bytes);
    g6_pad::OutTileWalker<out_tiles_h, out_tiles_w, out_lead1, in_tiles_h, in_tiles_w, in_lead0, in_lead1> walk(
        start_tile);

    const uint32_t num_blocks = (num_tiles + block_tiles - 1) / block_tiles;
    uint32_t remaining = num_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("load_block");
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_input_tiles, block_tiles);
        const uint32_t slot_base = get_write_ptr(cb_input_tiles);
        for (uint32_t k = 0; k < extent; ++k) {
            if (walk.source_bearing()) {
                if constexpr (rows_mode) {
                    // The input part of chunk wt: bytes [wt*chunk, min((wt+1)*chunk, in_row_bytes)) of the
                    // source row, landing at the start of the 64 B-aligned slot (chunk offsets are 64 B
                    // multiples, so src and dst share their alignment).
                    const uint32_t col = walk.wt * chunk_bytes;
                    const uint32_t left = in_row_bytes - col;
                    const uint32_t in_len = left < chunk_bytes ? left : chunk_bytes;
                    noc_async_read(src.get_noc_addr(walk.source_row(), col), slot_base + k * slot_bytes, in_len);
                } else {
                    noc_async_read(src.get_noc_addr(walk.source_page()), slot_base + k * slot_bytes, slot_bytes);
                }
            }
            walk.advance();
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
        remaining -= extent;
    }
}
