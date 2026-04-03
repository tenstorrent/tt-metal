// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BRISC dataflow kernel — reads x tiles into CB_IN0.
//
// Loop order matches the compute kernel: M_outer → K_outer.
// For each (M_block, K_block) one batch of M_block_tiles × K_block_tiles
// tiles is pushed into CB_IN0.  The CB naturally back-pressures this kernel
// until the compute kernel pops the previous batch (which only happens after
// the N-inner loop has finished using those x tiles for both gate and up).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────────────
    constexpr uint32_t M_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(4);

    // TensorAccessor compile-time metadata for x (interleaved DRAM).
    constexpr uint32_t ta_offset = 5;
    constexpr auto x_ta_args = TensorAccessorArgs<ta_offset>();

    // ── Runtime args ─────────────────────────────────────────────────────────
    const uint32_t x_addr = get_arg_val<uint32_t>(0);

    // Build accessor for x.
    const auto x_acc = TensorAccessor(x_ta_args, x_addr, in0_tile_size);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t in0_block_tiles = M_block_tiles * K_block_tiles;

    // x is laid out as a 2-D tile matrix: tile (m_tile, k_tile) has
    // linear tile_id = m_tile * K_tiles_total  +  k_tile.
    // We derive K_tiles_total = K_num_blocks * K_block_tiles.
    constexpr uint32_t K_tiles_total = K_num_blocks * K_block_tiles;

    for (uint32_t m = 0; m < M_num_blocks; m++) {
        uint32_t m_tile_base = m * M_block_tiles;

        for (uint32_t k = 0; k < K_num_blocks; k++) {
            uint32_t k_tile_base = k * K_block_tiles;

            cb_reserve_back(cb_in0, in0_block_tiles);
            uint32_t write_ptr = get_write_ptr(cb_in0);

            for (uint32_t mt = 0; mt < M_block_tiles; mt++) {
                for (uint32_t kt = 0; kt < K_block_tiles; kt++) {
                    uint32_t tile_id = (m_tile_base + mt) * K_tiles_total + (k_tile_base + kt);
                    noc_async_read_page(tile_id, x_acc, write_ptr);
                    write_ptr += in0_tile_size;
                }
            }
            noc_async_read_barrier();

            cb_push_back(cb_in0, in0_block_tiles);
        }
    }
}
