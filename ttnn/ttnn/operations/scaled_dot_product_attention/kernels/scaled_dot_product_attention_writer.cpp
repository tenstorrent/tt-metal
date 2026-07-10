// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for Flash-Attention scaled_dot_product_attention.
//
// Per Q-block, drains cb_out (q_cnt*Dt tiles, [q x D] row-major: for st(q): for dt)
// and writes them to the output tensor (B, H_q, S_q, D) tiled/interleaved.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t start_qb = get_arg_val<uint32_t>(1);
    uint32_t num_qb = get_arg_val<uint32_t>(2);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t q_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t q_blocks_per_bh = get_compile_time_arg_val(5);

    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t qi = 0; qi < num_qb; ++qi) {
        const uint32_t qb = start_qb + qi;
        const uint32_t bh = qb / q_blocks_per_bh;
        const uint32_t qci = qb % q_blocks_per_bh;
        const uint32_t b = bh / H_q;
        const uint32_t h_q = bh % H_q;

        const uint32_t q_row0 = qci * q_chunk_t;
        uint32_t q_cnt = q_chunk_t;
        if (q_row0 + q_cnt > Sq_t) {
            q_cnt = Sq_t - q_row0;
        }

        const uint32_t head_base = (b * H_q + h_q) * Sq_t;

        cb_wait_front(cb_out, q_cnt * Dt);
        uint32_t r = get_read_ptr(cb_out);
        uint32_t idx = 0;
        for (uint32_t st = q_row0; st < q_row0 + q_cnt; ++st) {
            for (uint32_t dt = 0; dt < Dt; ++dt) {
                uint32_t tid = (head_base + st) * Dt + dt;
                noc_async_write_tile(tid, out_acc, r + idx * tile_bytes);
                ++idx;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, q_cnt * Dt);
    }
}
