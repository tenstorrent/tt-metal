// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regime N (new_storage) writer, BRISC / NoC1.
//
// Per block: first every fill-only tile is written from cb_fill_tile (they depend on nothing, so
// they run in the shadow of the reader's block), then the block is awaited and every input-backed
// tile is written from its slot. Every output page is written exactly once. cb_fill_tile is
// self-owned scratch: filled once through its write pointer, never pushed or popped.
//
// Raw dataflow for the page writes: no kernel_lib helper wraps interleaved page writes for a block.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_pad_fill.hpp"
#include "g6_pad_new_storage_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_work_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);
    constexpr bool has_fill_only_tiles = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t geometry_ct_base = 6;
    using Walk = g6_pad::NewStorageWalk<geometry_ct_base>;
    constexpr auto out_args = TensorAccessorArgs<geometry_ct_base + g6_pad::kGeometryArgCount>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t core_has_fill_only = get_arg_val<uint32_t>(3);

    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    // build_fill_page: cb_fill_tile exists only when the call has fill-only tiles.
    [[maybe_unused]] uint32_t fill_addr = 0;
    if constexpr (has_fill_only_tiles) {
        fill_addr = get_write_ptr(cb_fill_tile);
        if (core_has_fill_only) {
            DeviceZoneScopedN("G6PAD_FILL_BUILD");
            g6_pad::fill_constant<elem_bytes>(cb_fill_tile, 0, tile_bytes, fill_bits);
        }
    }

    Walk walk;
    walk.start(core_work_begin);
    uint32_t block_first_page = core_work_begin;
    uint32_t tiles_left = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_extent = tiles_left < block_tiles ? tiles_left : block_tiles;

        // store_fill_tiles_block
        if constexpr (has_fill_only_tiles) {
            if (core_has_fill_only) {
                DeviceZoneScopedN("G6PAD_STORE_FILL");
                Walk fill_walk = walk;
                for (uint32_t i = 0; i < block_extent; ++i) {
                    if (!fill_walk.has_input()) {
                        noc_async_write_page(block_first_page + i, out_accessor, fill_addr);
                    }
                    fill_walk.advance();
                }
            }
        }

        // store_input_tiles_block
        {
            DeviceZoneScopedN("G6PAD_WRITE_WAIT");
            cb_wait_front(cb_work_tiles, block_tiles);
        }
        {
            DeviceZoneScopedN("G6PAD_STORE");
            const uint32_t block_base = get_read_ptr(cb_work_tiles);
            for (uint32_t i = 0; i < block_extent; ++i) {
                if (walk.has_input()) {
                    noc_async_write_page(block_first_page + i, out_accessor, block_base + i * tile_bytes);
                }
                walk.advance();
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_work_tiles, block_tiles);
        }
        block_first_page += block_extent;
        tiles_left -= block_extent;
    }

    // drain_writes (BB-03)
    {
        DeviceZoneScopedN("G6PAD_DRAIN");
        noc_async_write_barrier();
    }
}
