// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad regime N (new_storage) writer, BRISC / NoC1.
//
// Every output tile of the core's run is written once. Per block, the fill-only tiles go first
// from the fill page (they wait on nothing, so they hide the reader's latency), then the
// input-backed tiles from their cb_work_tiles slots once the reader pushes the block. One flush
// per block before the pop; one write barrier at the end.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "g6_pad_tile_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_work_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(5);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(6);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(7);
    constexpr uint32_t out_row_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t out_col_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t in_row_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t in_col_tiles = get_compile_time_arg_val(11);
    constexpr auto out_args = TensorAccessorArgs<12>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const bool core_has_fill_only = get_arg_val<uint32_t>(3) != 0;

    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    // cb_fill_tile exists only when the call has fill-only tiles; a core without any never
    // touches it. Self-owned scratch: written through the write pointer, never pushed or popped.
    uint32_t fill_base = 0;
    if (core_has_fill_only) {
        DeviceZoneScopedN("G6PAD_FILL_BUILD");
        fill_base = get_write_ptr(cb_fill_tile);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_base, tile_bytes, fill_bits);
    }

    using Walker =
        OutputTileWalker<out_lead1, in_lead0, in_lead1, out_row_tiles, out_col_tiles, in_row_tiles, in_col_tiles>;
    Walker walker;
    walker.init(core_work_begin);

    uint32_t block_begin = core_work_begin;
    uint32_t remaining = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        if (core_has_fill_only) {
            DeviceZoneScopedN("G6PAD_STORE_FILL");
            Walker tile = walker;
            for (uint32_t i = 0; i < extent; ++i) {
                if (!tile.has_input()) {
                    noc_async_write_page(block_begin + i, out_accessor, fill_base);
                }
                tile.advance();
            }
        }
        {
            DeviceZoneScopedN("G6PAD_STORE");
            {
                DeviceZoneScopedN("G6PAD_WRITE_WAIT");
                cb_wait_front(cb_work_tiles, block_tiles);
            }
            const uint32_t slot_base = get_read_ptr(cb_work_tiles);
            for (uint32_t i = 0; i < extent; ++i) {
                if (walker.has_input()) {
                    noc_async_write_page(block_begin + i, out_accessor, slot_base + i * tile_bytes);
                }
                walker.advance();
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_work_tiles, block_tiles);
        }
        block_begin += extent;
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("G6PAD_DRAIN");
        noc_async_write_barrier();
    }
}
