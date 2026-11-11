// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// One work unit is a (32-token block, stream copy): this core reads that block's x tile row, its key and value tile
// rows, and the gate vector of that copy, and owns the full D -- so the gate's three sums over D never leave the
// core and no core ever waits on another.
//
// The unit is read in two passes, because the gate must be known before value is scaled:
//
//   pass 1 (the sums): x, key, gate
//   pass 2 (the write): x again, value
//
// x is re-read rather than held in L1: its full D is 160 tiles at the prefill width, far more than a core keeps, and
// the second read is the same sequential DRAM stream as the first.
//
//   unit = block * hc + h          (streams dim 1 is the copy, dim 2 the token)
//   x[r, c]     = streams[0, h, block*32 + r, chunk*d_tiles + ...]
//   key[r, c]   = kv[0, 0, block*32 + r, h*D + ...]
//   value[r, c] = kv[0, 0, block*32 + r, hc*D + ...]     (the same row for every copy)
//   gate[c]     = gate_weight[0, h, :, ...]
//
// Inputs are TILE, so a run of consecutive tiles is one contiguous DRAM run: each read is a multi-tile transfer.
void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_v = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(5);  // tiles of D per chunk
    constexpr uint32_t d_tiles = get_compile_time_arg_val(6);      // the full D, in tiles
    constexpr uint32_t hc = get_compile_time_arg_val(7);
    constexpr uint32_t blocks = get_compile_time_arg_val(8);  // tile rows per stream copy

    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto kv_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gate_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t gate_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);

    const auto x = TensorAccessor(x_args, x_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto gate = TensorAccessor(gate_args, gate_addr);

    // REDUCE scaler: 1.0 in the first row of each fp32 face. Pushed before any unit, because compute waits on it
    // even on a core that owns no units.
    cb_reserve_back(cb_scaler, 1);
    {
        volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < 1024; ++i) {
            scaler[i] = 0;
        }
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t j = 0; j < 16; ++j) {
                scaler[face * 256 + j] = 0x3F800000;  // 1.0f
            }
        }
    }
    cb_push_back(cb_scaler, 1);

    constexpr uint32_t bf16_tile_bytes = 32 * 32 * 2;
    constexpr uint32_t fp32_tile_bytes = 32 * 32 * 4;
    const uint32_t num_chunks = d_tiles / chunk_tiles;

    // Copies `chunk_tiles` consecutive tiles from `first` into `cb`.
    auto read_run = [&](const auto& accessor, uint32_t first, uint32_t cb, uint32_t tile_bytes) {
        cb_reserve_back(cb, chunk_tiles);
        const uint32_t dst = get_write_ptr(cb);
        for (uint32_t s = 0; s < chunk_tiles; ++s) {
            noc_async_read(accessor.get_noc_addr(first + s), dst + s * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, chunk_tiles);
    };

    for (uint32_t u = unit_start; u < unit_start + units; ++u) {
        const uint32_t block = u / hc;
        const uint32_t h = u - block * hc;
        // A stream's tile rows are (copy, block): copy h, block b is row h * blocks + b. kv is [1, 1, T, (hc+1)*D],
        // so its tile rows are just the blocks, each (hc + 1) * d_tiles tiles wide.
        const uint32_t x_row = (h * blocks + block) * d_tiles;
        const uint32_t kv_row = block * (hc + 1) * d_tiles;

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t c0 = chunk * chunk_tiles;
            read_run(x, x_row + c0, cb_x, bf16_tile_bytes);
            read_run(kv, kv_row + h * d_tiles + c0, cb_k, bf16_tile_bytes);
            read_run(gate, h * d_tiles + c0, cb_g, fp32_tile_bytes);
        }
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t c0 = chunk * chunk_tiles;
            read_run(x, x_row + c0, cb_x, bf16_tile_bytes);
            read_run(kv, kv_row + hc * d_tiles + c0, cb_v, bf16_tile_bytes);
        }
    }
}
