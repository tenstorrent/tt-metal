// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage_tiles reader (RISCV_1 / NOC0).
// N2 load_block: reserve block_tiles slots; for every source-bearing output tile k of the block, read
// its single input tile into slot k; ONE read barrier; push block_tiles (nominal -- fill-only slots and
// a ragged last block are simply left unwritten, so every ring cycle closes at capacity).
// Fill-only output tiles are never read (the writer sources them from cb_fill_tile).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_pad_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_tiles_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_tiles_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(5);
    constexpr uint32_t in_tiles_h = get_compile_time_arg_val(6);
    constexpr uint32_t in_tiles_w = get_compile_time_arg_val(7);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(8);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(9);
    constexpr auto src_args = TensorAccessorArgs<10>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
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
                noc_async_read(src.get_noc_addr(walk.source_page()), slot_base + k * tile_bytes, tile_bytes);
            }
            walk.advance();
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
        remaining -= extent;
    }
}
