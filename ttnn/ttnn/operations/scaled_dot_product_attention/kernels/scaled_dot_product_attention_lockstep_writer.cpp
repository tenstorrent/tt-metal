// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the Q-outer lockstep multicast SDPA variant. The (reused, baseline)
// compute kernel emits one q-chunk's normalized output to cb_out per wave; this
// writer drains it and writes to output[b, h, qc(col,k)*Sq_chunk_t block, :],
// where qc(col,k) = min(k*cores_per_row + col, n_q_chunks-1) matches the reader's
// clamp-padded wave assignment. The clamped (padding) waves write the last
// q-chunk's output again with the same value — harmless.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_out = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(3);
    constexpr bool ablate_writer = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t num_waves = get_compile_time_arg_val(5);
    constexpr uint32_t cores_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);
    const uint32_t h = get_arg_val<uint32_t>(2);
    const uint32_t col = get_arg_val<uint32_t>(3);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto acc = TensorAccessor(dst_args, out_addr, tile_bytes);

    constexpr uint32_t q_tiles = Sq_chunk_t * Dt;
    const uint32_t head_base = (b * H + h) * Sq_t;

    for (uint32_t k = 0; k < num_waves; ++k) {
        uint32_t qc = k * cores_per_row + col;
        if (qc >= n_q_chunks) {
            qc = n_q_chunks - 1;
        }
        const uint32_t sq_off = qc * Sq_chunk_t;
        cb_wait_front(cb_out, q_tiles);
        if constexpr (!ablate_writer) {
            uint32_t rptr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < q_tiles; ++t) {
                const uint32_t sq_g = sq_off + (t / Dt);
                const uint32_t d = t % Dt;
                noc_async_write_tile((head_base + sq_g) * Dt + d, acc, rptr);
                rptr += tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, q_tiles);
    }
}
