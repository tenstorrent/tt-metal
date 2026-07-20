// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize GENERAL reader (NCRISC / NoC0) — cross-core sharded / crossover path.
//
// Reads the ROW_MAJOR sticks for this core's assigned output tile-rows out of
// ANY input placement (interleaved DRAM/L1 OR L1-sharded) via TensorAccessor,
// which resolves each logical page to its physical location (DRAM bank or a
// remote L1 shard bank on another core). This is the native default-factory
// data-access model: work is split across the compute grid and each core pulls
// its own input rows via NoC — no cross-core semaphores.
//
// Wide-W chunking (Refinement 2d, memory-budget lever #1): instead of assembling
// the full Wt-wide row per block (CB = 2*Wt*tile, L1-unbounded when W is large),
// an outer chunk loop processes `Wt_chunk` (constant) output tile-columns per
// pass. For chunk `k` each stick contributes only the byte range
// `[k*chunk_width_bytes, (k+1)*chunk_width_bytes)` of the full logical row, so
// the CB is 2*Wt_chunk*tile — CONSTANT in W. Loop order (chunk-outer, block-inner)
// matches the reused tilize_compute.cpp / tilize_writer.cpp, which already take
// num_chunks / Wt_chunk CT args.
//
// The npr subtlety: a WIDTH/BLOCK-sharded input splits each logical row into
// `npr = ceil(W / shard_width)` accessor pages of `shard_page_bytes` each. An
// output tile-column chunk's contiguous byte range may span a SUBSET of those
// shard pages (and start/end mid-page), so per stick we walk the overlapping
// shard pages: page index = abs_byte / shard_page_bytes, in-page offset =
// abs_byte % shard_page_bytes. For interleaved / HEIGHT-sharded input npr==1 and
// shard_page_bytes==row_bytes, so this degenerates to a single per-stick read at
// byte offset k*chunk_width_bytes (the interleaved-reader chunking pattern).
//
// Helper deviation (unchanged from 2c): dataflow_kernel_lib::read_sticks_for_tilize
// is stick-indexed (exactly ONE accessor page per logical row) and cannot
// assemble a full-width stick from a WIDTH-split sharded input where one logical
// row spans npr pages. This kernel adds the per-row npr chunk loop the helper
// lacks; the L1 layout it produces (32 sticks, each strided by chunk_width_bytes)
// is byte-identical to the helper's TILE-granularity output, so the downstream
// compute_kernel_lib::tilize consumes it unchanged.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t Wt_chunk = get_compile_time_arg_val(0);           // output tile-cols per chunk (constant)
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);         // Wt / Wt_chunk
    constexpr uint32_t chunk_width_bytes = get_compile_time_arg_val(2);  // Wt_chunk * tile_row_bytes (CB stick stride)
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);          // full logical row bytes (W * elem)
    constexpr uint32_t shard_page_bytes = get_compile_time_arg_val(4);   // input accessor page bytes (shard-width row)
    constexpr uint32_t npr = get_compile_time_arg_val(5);                // accessor pages per logical row
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);  // logical row offset (start_tile_row * 32)
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // count * 32

    const auto accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t TILE_H = 32;
    const uint32_t num_blocks = num_rows / TILE_H;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t chunk_col_off = chunk * chunk_width_bytes;
        uint32_t cur_width = chunk_width_bytes;
        if (chunk_col_off + cur_width > row_bytes) {
            cur_width = row_bytes - chunk_col_off;  // last chunk clip (no-op when Wt_chunk divides Wt)
        }
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_rm_in, Wt_chunk);
            const uint32_t l1_base = get_write_ptr(cb_rm_in);
            for (uint32_t r = 0; r < TILE_H; ++r) {
                const uint32_t row = start_row + block * TILE_H + r;
                const uint32_t l1_stick = l1_base + r * chunk_width_bytes;
                uint32_t written = 0;
                while (written < cur_width) {
                    const uint32_t abs_byte = chunk_col_off + written;
                    const uint32_t shard_page = abs_byte / shard_page_bytes;
                    const uint32_t page_off = abs_byte - shard_page * shard_page_bytes;
                    uint32_t take = shard_page_bytes - page_off;
                    if (take > cur_width - written) {
                        take = cur_width - written;
                    }
                    noc_async_read(accessor.get_noc_addr(row * npr + shard_page, page_off), l1_stick + written, take);
                    written += take;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm_in, Wt_chunk);
        }
    }
}
