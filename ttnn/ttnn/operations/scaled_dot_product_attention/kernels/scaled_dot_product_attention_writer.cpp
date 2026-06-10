// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention writer (BRISC / NoC1).
//
// Per work unit (b, h, q_chunk): streams cur_cq * Dt output tiles from
// cb_out_tiles (tile-row-major (r, d) order — matches the compute's
// TileRowMajor pack / streaming chain pack) into the interleaved output.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t c_q = get_compile_time_arg_val(3);
    constexpr uint32_t Nq = get_compile_time_arg_val(4);
    constexpr uint32_t c_q_last = get_compile_time_arg_val(5);

    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    if (num_units == 0) {
        return;
    }

    constexpr uint32_t cb_out_tiles = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out_tiles);
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t bh = unit / Nq;
        const uint32_t qc = unit % Nq;
        const uint32_t cur_cq = (qc == Nq - 1) ? c_q_last : c_q;
        const uint32_t q_row0 = qc * c_q;
        const uint32_t head_base = bh * Sq_t * Dt;

        // Batch the whole chunk: one wait, all writes in flight, one barrier,
        // one pop (CB holds 2 * c_q * Dt pages, so a full chunk always fits).
        const uint32_t chunk_tiles = cur_cq * Dt;
        cb_wait_front(cb_out_tiles, chunk_tiles);
        uint32_t l1_addr = get_read_ptr(cb_out_tiles);
        for (uint32_t r = 0; r < cur_cq; ++r) {
            const uint32_t row_base = head_base + (q_row0 + r) * Dt;
            for (uint32_t d = 0; d < Dt; ++d) {
                noc_async_write_tile(row_base + d, out_accessor, l1_addr);
                l1_addr += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_tiles, chunk_tiles);
    }
}
