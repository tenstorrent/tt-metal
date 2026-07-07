// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention — Writer Kernel (BRISC)
// Reads output tiles from the output CB and writes them to DRAM.
//
// CT arg layout:
//   [0] S_q_t       — S_q in tiles
//   [1] D_t         — head dim in tiles
//   [2] H           — num heads (for tile offset computation)
//   [3] B_q          — Q block size in tiles
//   [4] num_q_blocks — number of Q blocks
//   [5..] Output TensorAccessorArgs
//
// RT arg layout:
//   [0] num_work_units — number of (B,H) pairs this core processes
//   [1] output_addr
//   [2] start_tile_id_0 — first tile id for work unit 0
//   [3] start_tile_id_1 — first tile id for work unit 1 (if present)
//   ...

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t S_q_t = get_compile_time_arg_val(0);
    const uint32_t D_t = get_compile_time_arg_val(1);
    const uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t B_q = get_compile_time_arg_val(3);
    const uint32_t num_q_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t D_CHUNK = get_compile_time_arg_val(5);
    const uint32_t num_d_chunks = get_compile_time_arg_val(6);

    constexpr auto out_args = TensorAccessorArgs<7>();

    // RT args: [num_work_units, output_addr, start_tile_id_0, ...]
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_output = 16;

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    uint32_t rt_arg_idx = 2;  // start_tile_ids start at RT arg index 2
    for (uint32_t wu = 0; wu < num_work_units; wu++) {
        const uint32_t start_tile_id = get_arg_val<uint32_t>(rt_arg_idx++);

        for (uint32_t qb = 0; qb < num_q_blocks; qb++) {
            // D_CHUNK outer loop: writer writes D_CHUNK tiles per Q block per chunk.
            // The compute kernel pushes D_CHUNK output tiles per (qb, dc) iteration.
            // Tile offset advances by D_CHUNK per chunk within each Q block row.
            for (uint32_t dc = 0; dc < num_d_chunks; dc++) {
                uint32_t tile_offset = start_tile_id + qb * B_q * D_t + dc * D_CHUNK;
                for (uint32_t sq = 0; sq < B_q; sq++) {
                    if (qb * B_q + sq >= S_q_t) {
                        break;
                    }
                    for (uint32_t d = 0; d < D_CHUNK; d++) {
                        cb_wait_front(cb_output, 1);
                        uint32_t l1_read_addr = get_read_ptr(cb_output);
                        noc_async_write_tile(tile_offset, out_accessor, l1_read_addr);
                        noc_async_write_barrier();
                        cb_pop_front(cb_output, 1);
                        tile_offset++;
                    }
                    // Advance to next Q row (skip remaining D_t - D_CHUNK tiles
                    // in this row, then add D_CHUNK for the current chunk offset)
                    if (sq < B_q - 1) {
                        tile_offset += (D_t - D_CHUNK) + dc * D_CHUNK;
                    }
                }
            }
        }
    }
}
