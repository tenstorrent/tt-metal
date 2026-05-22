// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Drains cb_out (the final per-core output block of the down matmul, shape
// per_core_M * per_core_N_d tiles) and writes it into the DRAM-interleaved
// output tensor at this core's (mt, nt_d) position. The compute kernel packs
// the output one subblock at a time, so we wait_front on subblock-sized
// increments to keep the CB protocol in sync.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(5);

    constexpr uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    constexpr uint32_t out_block_num_tiles = per_core_M * per_core_N_d;
    // The compute kernel packs `in0_num_subblocks * in1_num_subblocks =
    // (per_core_M/out_subblock_h) * (per_core_N_d/out_subblock_w)` subblocks.
    static_assert(out_block_num_tiles % out_subblock_num_tiles == 0);
    constexpr uint32_t num_subblocks = out_block_num_tiles / out_subblock_num_tiles;
    constexpr uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    constexpr uint32_t in1_num_subblocks = per_core_N_d / out_subblock_w;

    constexpr uint32_t out_accessor_offset = 6;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    // Compute the (row, col) of the first tile this core writes.
    const uint32_t row0 = chunk_start_tile_row + my_mt * per_core_M;
    const uint32_t col0 = my_nt_d * per_core_N_d;

    // The compute kernel packs subblocks in row-major order over (in0_subblock,
    // in1_subblock). Each subblock is (out_subblock_h tile-rows) x
    // (out_subblock_w tile-cols). We mirror that ordering on the write side.
    for (uint32_t sb_m = 0; sb_m < in0_num_subblocks; ++sb_m) {
        for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
            cb_wait_front(cb_out, out_subblock_num_tiles);
            uint32_t l1_read = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < out_subblock_h; ++i) {
                for (uint32_t j = 0; j < out_subblock_w; ++j) {
                    const uint32_t row = row0 + sb_m * out_subblock_h + i;
                    const uint32_t col = col0 + sb_n * out_subblock_w + j;
                    const uint32_t tile_idx = row * N_down_tiles_full + col;
                    noc_async_write_tile(tile_idx, out_acc, l1_read);
                    l1_read += out_tile_bytes;
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, out_subblock_num_tiles);
        }
    }
}
