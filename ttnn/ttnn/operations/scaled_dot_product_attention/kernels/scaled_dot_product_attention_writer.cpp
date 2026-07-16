// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for scaled_dot_product_attention. Drains cb_out (the normalized
// attention output for one q-chunk, Sq_chunk_t x Dt tiles) to DRAM at
// output[b, h, q-chunk-block, :]. Same work-unit decode as the reader.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_out = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(5);

    constexpr auto dst_args = TensorAccessorArgs<6>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_wu = get_arg_val<uint32_t>(1);
    const uint32_t num_wu = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto acc = TensorAccessor(dst_args, out_addr, tile_bytes);

    const uint32_t HQ = H * n_q_chunks;

    for (uint32_t wi = 0; wi < num_wu; ++wi) {
        const uint32_t w = start_wu + wi;
        const uint32_t b = w / HQ;
        const uint32_t r = w % HQ;
        const uint32_t h = r / n_q_chunks;
        const uint32_t qc = r % n_q_chunks;

        const uint32_t base = (b * H + h) * Sq_t;
        for (uint32_t sq = 0; sq < Sq_chunk_t; ++sq) {
            const uint32_t sq_g = qc * Sq_chunk_t + sq;
            for (uint32_t d = 0; d < Dt; ++d) {
                cb_wait_front(cb_out, 1);
                noc_async_write_tile((base + sq_g) * Dt + d, acc, get_read_ptr(cb_out));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
