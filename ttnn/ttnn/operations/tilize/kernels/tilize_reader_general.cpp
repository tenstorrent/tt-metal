// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize GENERAL reader (NCRISC / NoC0) — cross-core sharded / crossover path.
//
// Reads the full-width ROW_MAJOR sticks for this core's assigned output
// tile-rows out of ANY input placement (interleaved DRAM/L1 OR L1-sharded) via
// TensorAccessor, which resolves each logical page to its physical location
// (DRAM bank or a remote L1 shard bank on another core). This is the native
// default-factory data-access model: work is split across the compute grid and
// each core pulls its own input rows via NoC — no cross-core semaphores.
//
// Helper deviation: dataflow_kernel_lib::read_sticks_for_tilize is stick-indexed
// (exactly ONE accessor page per logical row) and therefore cannot assemble a
// full-width stick from a WIDTH-split sharded input, where one logical row spans
// `npr = ceil(W / shard_width)` accessor pages. This kernel adds the per-row
// chunk loop the helper lacks; the L1 layout it produces (32 sticks, each padded
// to Wt*tile_row_bytes) is byte-identical to the helper's TILE-granularity
// output, so the downstream compute_kernel_lib::tilize consumes it unchanged.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);           // full width in tiles (W / 32)
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);    // full logical row bytes (W * elem)
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(2);  // input accessor page bytes (shard-width row)
    constexpr uint32_t npr = get_compile_time_arg_val(3);          // accessor pages per logical row
    constexpr auto src_args = TensorAccessorArgs<4>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);  // logical row offset (start_tile_row * 32)
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // count * 32

    const auto accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t TILE_H = 32;
    const uint32_t num_blocks = num_rows / TILE_H;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_addr = get_write_ptr(cb_rm_in);
        for (uint32_t r = 0; r < TILE_H; ++r) {
            const uint32_t row = start_row + block * TILE_H + r;
            uint32_t written = 0;
            for (uint32_t c = 0; c < npr; ++c) {
                uint32_t take = chunk_bytes;
                if (written + take > row_bytes) {
                    take = row_bytes - written;  // last width-chunk: valid bytes only (pad clipped)
                }
                noc_async_read(accessor.get_noc_addr(row * npr + c), l1_addr + written, take);
                written += take;
            }
            l1_addr += row_bytes;  // = Wt * tile_row_bytes (W is tile-aligned), matches tilize's stick stride
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm_in, Wt);
    }
}
