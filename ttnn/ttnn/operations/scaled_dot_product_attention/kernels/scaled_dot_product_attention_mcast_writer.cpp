// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the NoC-multicast SDPA variant. The KV-outer compute emits its
// q_cnt owned sub-chunks' normalized outputs to cb_out one after another AFTER
// the KV loop; this writer drains them in that order to
// output[b, h, (q_start+s)*Sq_chunk_t block, :]. TILE layout, row-major (sq, d).

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
    // MEASUREMENT-ONLY (env SDPA_MCAST_ABLATE_WRITER): skip the output DRAM writes,
    // keeping cb_out wait/pop intact — writer twin of the reader floor probe.
    constexpr bool ablate_writer = get_compile_time_arg_val(4) != 0;
    constexpr auto dst_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);
    const uint32_t h = get_arg_val<uint32_t>(2);
    const uint32_t q_start = get_arg_val<uint32_t>(3);
    const uint32_t q_cnt = get_arg_val<uint32_t>(4);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto acc = TensorAccessor(dst_args, out_addr, tile_bytes);

    constexpr uint32_t q_tiles = Sq_chunk_t * Dt;
    const uint32_t head_base = (b * H + h) * Sq_t;  // first Q tile-row of this (b,h)

    for (uint32_t s = 0; s < q_cnt; ++s) {
        const uint32_t sq_off = (q_start + s) * Sq_chunk_t;
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
