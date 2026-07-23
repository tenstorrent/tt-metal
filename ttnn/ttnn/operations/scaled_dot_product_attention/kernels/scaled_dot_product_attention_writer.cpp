// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-attention writer. Drains cb_out (one normalized Q-block at a time,
// natural (SQ_CHUNK_T, DHT) tile grid) to the output tensor. Mirrors the
// reader's per-core Q-block work slice so output tile ids line up.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_Q = get_compile_time_arg_val(1);
    constexpr uint32_t SQT = get_compile_time_arg_val(2);
    constexpr uint32_t DHT = get_compile_time_arg_val(3);
    constexpr uint32_t SQ_CHUNK_T = get_compile_time_arg_val(4);
    constexpr uint32_t Q_NUM_CHUNKS = get_compile_time_arg_val(5);
    constexpr uint32_t USE_MCAST = get_compile_time_arg_val(6);
    constexpr uint32_t GC = get_compile_time_arg_val(7);

    constexpr auto out_args = TensorAccessorArgs<8>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_start = get_arg_val<uint32_t>(1);
    const uint32_t q_count = get_arg_val<uint32_t>(2);
    // Refinement 3 mcast work-mapping (per-core; used only when USE_MCAST):
    const uint32_t row_y = get_arg_val<uint32_t>(3);
    const uint32_t col_x = get_arg_val<uint32_t>(4);
    const uint32_t rounds = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t out_block_tiles = SQ_CHUNK_T * DHT;

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    // Number of output Q-blocks this core drains, and the per-slot (nb,nq,q_chunk)
    // mapping. Non-mcast: the flat work slice [q_start, q_start+q_count). Mcast:
    // `rounds` slots for one head-row; slot rd -> q_chunk = rd*GC+col_x (or 0 for
    // a dummy slot, whose output is a bit-identical redundant write of Q-chunk 0).
    const uint32_t n_blocks = USE_MCAST ? rounds : q_count;

    for (uint32_t idx = 0; idx < n_blocks; ++idx) {
        uint32_t q_chunk, nq, nb;
        if constexpr (USE_MCAST) {
            const uint32_t qc_raw = idx * GC + col_x;
            q_chunk = (qc_raw < Q_NUM_CHUNKS) ? qc_raw : 0u;
            nq = row_y % H_Q;
            nb = row_y / H_Q;
        } else {
            const uint32_t i = q_start + idx;
            q_chunk = i % Q_NUM_CHUNKS;
            const uint32_t t = i / Q_NUM_CHUNKS;
            nq = t % H_Q;
            nb = t / H_Q;
        }

        cb_wait_front(cb_out, out_block_tiles);
        uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t o_base = (nb * H_Q + nq) * SQT;
        for (uint32_t st = 0; st < SQ_CHUNK_T; ++st) {
            const uint32_t s_tile = q_chunk * SQ_CHUNK_T + st;
            for (uint32_t dt = 0; dt < DHT; ++dt) {
                noc_async_write_tile((o_base + s_tile) * DHT + dt, out_acc, l1);
                l1 += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_block_tiles);
    }
}
