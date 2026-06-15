// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Single responsibility: pop `cb_out` (the down matmul's per-core final
// block, packed one subblock at a time) and write tiles to the DRAM-
// interleaved output tensor at this core's (mt, nt_d) tile region, looped
// over `effective_chunks` chunks. Reads the device-side counts/idx
// scratch CBs to compute effective_chunks. Output writes start at tile
// row 0 — the FFN op writes to a per-expert output tensor; ttnn::insert
// handles placement into any shared destination buffer. The activated
// tiles are distributed across the M-row by the READER via L1 multicast —
// there is no DRAM scratch round-trip or cross-core barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);
    // NEW: device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(15);
    // M_tiles_full: total tile-row count of the output tensor. When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those writes here so we
    // don't OOB-write the output buffer.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;

    constexpr uint32_t out_accessor_offset = 17;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    // Wait for the reader's counts/idx push and compute effective_chunks =
    // ceil(count / chunk_M_tiles). The writer drains cb_out per chunk;
    // bounding the loop here is required because the reader and compute
    // bound theirs too — without this, the writer would wait forever on
    // cb_out for chunks the compute never pushes.
    cb_wait_front(cb_counts_scratch, 1);
    cb_wait_front(cb_idx_scratch, 1);
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_counts_scratch));
    const uint32_t idx_l1 = get_read_ptr(cb_idx_scratch);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    const uint32_t count_tiles = (count_value + 31) / 32;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;

    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
        const uint32_t col0 = my_nt_d * per_core_N_d;
        for (uint32_t sb_m = 0; sb_m < d_in1_num_subblocks_M; ++sb_m) {
            for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                cb_wait_front(cb_out, d_out_subblock_num_tiles);
                uint32_t l1_read = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                    for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                        const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                        const uint32_t col = col0 + sb_n * d_out_subblock_w + j;
                        // Skip OOB writes:
                        //   * col >= N_down_tiles_full: GRID_X=11 ceil_div
                        //     produces phantom output cols past actual N.
                        //   * row >= M_tiles_full: ceil_div of M produces a
                        //     last-chunk tail past actual M when M_tiles_full
                        //     doesn't divide chunk_M_tiles. The matmul still
                        //     runs on those rows (reader zero-fills input),
                        //     but the output isn't part of the result tensor.
                        // Bounds:
                        //   * col < N_down_tiles_full: GRID_X=11 ceil_div
                        //     produces phantom output cols past actual N.
                        //   * row < M_tiles_full: ceil_div of M produces a
                        //     last-chunk tail past actual M when
                        //     M_tiles_full doesn't divide chunk_M_tiles.
                        //   * row < count_tiles: the last chunk's per_core_M
                        //     rows extend past count_tiles when count_tiles
                        //     is not chunk-aligned.
                        if (col < N_down_tiles_full && row < M_tiles_full && row < count_tiles) {
                            const uint32_t tile_idx = row * N_down_tiles_full + col;
                            noc_async_write_page(tile_idx, out_acc, l1_read);
                        }
                        l1_read += out_tile_bytes;
                    }
                }
                // Wait for the writes to LEAVE this core (departed sender);
                // doesn't wait for the DRAM round-trip. Safe to reuse the L1
                // slot now — the NoC has captured the data. ~10x faster than
                // noc_async_write_barrier per subblock at small per_core_M.
                noc_async_writes_flushed();
                cb_pop_front(cb_out, d_out_subblock_num_tiles);
            }
        }
    }
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc_async_write_barrier();
}
