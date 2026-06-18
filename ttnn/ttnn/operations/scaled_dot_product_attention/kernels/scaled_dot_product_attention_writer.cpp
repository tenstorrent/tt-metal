// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for Flash-Attention SDPA.
//
// Per work item (b, h_q, qb): drain one [B_q, vDHt] output block from cb_out
// (tile-row-major) and write it to the output tensor O (B, H_q, S_q, D) at tile
// (st = qb*B_q + q, dt = d). Output tile grid matches Q's layout.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t start_work = get_arg_val<uint32_t>(0);
    uint32_t num_work = get_arg_val<uint32_t>(1);
    uint32_t out_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t H_q = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);  // == vDHt
    constexpr uint32_t B_q = get_compile_time_arg_val(3);
    constexpr uint32_t n_q = get_compile_time_arg_val(4);

    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t w = start_work; w < start_work + num_work; ++w) {
        const uint32_t qb = w % n_q;
        const uint32_t r = w / n_q;
        const uint32_t h_q = r % H_q;
        const uint32_t b = r / H_q;

        const uint32_t head_base = (b * H_q + h_q) * Sq_t;

        for (uint32_t q = 0; q < B_q; ++q) {
            const uint32_t row_page = (head_base + qb * B_q + q) * DHt;
            for (uint32_t d = 0; d < DHt; ++d) {
                cb_wait_front(cb_out, 1);
                const uint32_t rptr = get_read_ptr(cb_out);
                noc_async_write_tile(row_page + d, out_acc, rptr);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
