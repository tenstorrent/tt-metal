// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for scaled_dot_product_attention (Flash Attention).
//
// Drains cb_out (final normalized output O / l_i) from L1 to DRAM.
// The compute kernel produces B_q_t * D_t output tiles per Q-block, per work unit.
// The writer writes tiles to the correct DRAM page indices based on (b, h, qb).
//
// Output tensor: (B, H_q, S_q, D) in TILE_LAYOUT.
// Page index for tile (b, h, sq, d) = b*H_q*S_q_tiles*D_t + h*S_q_tiles*D_t + sq*D_t + d
//
// CT args: [B_q_t, D_t, num_q_blocks, ...TensorAccessorArgs(output)]
// RT args: [output_addr, num_work_units, S_q_tiles, H_q,
//           b0, h0, b1, h1, ...]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);
    constexpr uint32_t D_t = get_compile_time_arg_val(1);
    constexpr uint32_t num_q_blocks = get_compile_time_arg_val(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_17;

    uint32_t rt_idx = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t num_work_units = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_q_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t H_q = get_arg_val<uint32_t>(rt_idx++);

    // Read (b, h) pairs
    uint32_t work_b[16], work_h[16];
    for (uint32_t i = 0; i < num_work_units; ++i) {
        work_b[i] = get_arg_val<uint32_t>(rt_idx++);
        work_h[i] = get_arg_val<uint32_t>(rt_idx++);
    }

    constexpr auto dst_args = TensorAccessorArgs<3>();
    uint32_t tile_bytes = get_tile_size(cb_out);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    constexpr uint32_t num_o_tiles_per_block = B_q_t * D_t;

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t b = work_b[wu];
        uint32_t h = work_h[wu];

        // Base output page index for this (b, h) pair
        uint32_t bh_base = b * H_q * S_q_tiles * D_t + h * S_q_tiles * D_t;

        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            uint32_t q_row_start = qb * B_q_t;

            // Read tiles from cb_out and write to DRAM
            // Tiles are in row-major order: for r=0..B_q_t-1, for d=0..D_t-1
            for (uint32_t r = 0; r < B_q_t; ++r) {
                for (uint32_t d = 0; d < D_t; ++d) {
                    cb_wait_front(cb_out, 1);
                    uint32_t l1_read_addr = get_read_ptr(cb_out);
                    uint32_t out_tile_idx = bh_base + (q_row_start + r) * D_t + d;
                    noc_async_write_tile(out_tile_idx, accessor, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, 1);
                }
            }
        }
    }
}
