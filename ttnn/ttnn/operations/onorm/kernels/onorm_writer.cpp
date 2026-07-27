// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// onorm writer (BRISC / NoC1).
//
// Drains finished flat token-major output tiles from cb_out_tiles and writes
// them to their interleaved DRAM pages.  Symmetric to the reader: writes go out
// in `dm_block_tiles`-sized groups with ONE noc_async_write_barrier per group,
// so up to `dm_block_tiles` writes are in flight and the transfers pipeline.
// The group size is the same DM_BLOCK_TILES knob the reader uses (a compile-time
// arg on both sides) — this is the writer half of the dataflow lever.
//
// The writer deliberately performs NO reads (it does not fetch `gate`, even
// though that would balance per-core byte counts): reads issued on NoC1 measured
// ~4.8x slower than on NoC0.  See op_design.md §6.
//
// HELPER USAGE: no kernel_lib helper wraps plain interleaved-DRAM tile
// streaming, so TensorAccessor + noc_async_write is raw by necessity.

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    // CB slot map — injected as a preprocessor define from the ONE host-side
    // source of truth (`_CB_SLOTS` in onorm_program_descriptor.py).
    constexpr uint32_t cb_out_tiles = ONORM_CB_OUT_TILES;

    // --- Blocking Model parameters (compile-time; one source of truth on host) ---
    constexpr uint32_t flat_tiles = get_compile_time_arg_val(0);           // FLAT / TILE_W
    constexpr uint32_t tile_rows_per_block = get_compile_time_arg_val(1);  // TOKENS_PER_BLOCK / TILE_H
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(2);     // ceil(T / TOKENS_PER_BLOCK)
    constexpr uint32_t token_tile_rows = get_compile_time_arg_val(3);      // Tt = ceil(T / TILE_H)
    constexpr uint32_t dm_block_tiles = get_compile_time_arg_val(4);       // DM_BLOCK_TILES
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);

    constexpr auto out_args = TensorAccessorArgs<6>();

    // Derived — never a restated literal.
    constexpr uint32_t flat_tiles_per_block = tile_rows_per_block * flat_tiles;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t bi = start_block + blk;
        const uint32_t b = bi / blocks_per_batch;
        const uint32_t r = bi % blocks_per_batch;

        // Output shares `gate`'s (T, FLAT) tiling, so the token axis is tile-padded.
        const uint32_t first_tile = (b * token_tile_rows + r * tile_rows_per_block) * flat_tiles;

        MaybeDeviceZoneScope("onorm_write_out");
        uint32_t done = 0;
        while (done < flat_tiles_per_block) {
            const uint32_t remaining = flat_tiles_per_block - done;
            const uint32_t n = remaining < dm_block_tiles ? remaining : dm_block_tiles;
            cb_wait_front(cb_out_tiles, n);
            const uint32_t l1_read_addr = get_read_ptr(cb_out_tiles);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(first_tile + done + i), page_bytes);
            }
            noc_async_write_barrier();  // ONE barrier for `n` writes
            cb_pop_front(cb_out_tiles, n);
            done += n;
        }
    }
}
