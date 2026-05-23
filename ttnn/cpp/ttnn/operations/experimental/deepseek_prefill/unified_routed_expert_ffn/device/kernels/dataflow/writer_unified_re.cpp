// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Single responsibility after the L1-mcast refactor: pop `cb_out` (the down
// matmul's per-core final block, packed one subblock at a time) and write
// tiles to the DRAM-interleaved output tensor at this core's (mt, nt_d) tile
// region. Looped over chunks.
//
// The previous phase-3 drain to DRAM scratch + cross-core barrier is gone —
// activated tiles are now distributed across the M-row by the READER via
// L1 multicast (one sender per phase-4 K-block).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    // Runtime args 1..12 are legacy from the scratch-barrier era. Ignored here
    // but kept in the layout so program_factory does not need to renumber
    // existing args yet.
    const uint32_t my_mt = get_arg_val<uint32_t>(8);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;

    constexpr uint32_t out_accessor_offset = 13;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
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
                        const uint32_t tile_idx = row * N_down_tiles_full + col;
                        noc_async_write_tile(tile_idx, out_acc, l1_read);
                        l1_read += out_tile_bytes;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, d_out_subblock_num_tiles);
            }
        }
    }
}
