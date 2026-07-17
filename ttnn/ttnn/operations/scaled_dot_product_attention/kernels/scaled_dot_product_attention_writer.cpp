// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for scaled_dot_product_attention. Drains cb_out (the normalized
// attention output for one q-chunk, Sq_chunk_t x Dt tiles) to DRAM at
// output[b, h, q-chunk-block, :]. Same work-unit decode as the reader.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_out = 16;

// R3 (data-movement, writer twin of the reader batching): drain a block of tiles
// with ONE barrier instead of one write + barrier + pop per tile. `batch` is the
// compile-time full-slot predicate (no partial q-chunk), so the multi-page linear
// read-pointer walk stays contiguous in the OUT_DEPTH-slot ring.
//
// `ablate` (MEASUREMENT-ONLY, /perf-measure classify-the-bound, writer twin of the
// reader stub): when true, SKIP the noc_async_write_tile + barrier but KEEP
// cb_wait_front/cb_pop_front — output DRAM bytes moved drop to zero while the
// compute->writer CB counts are unchanged. A flat wall-time with writes stubbed
// proves the output writes are hidden behind compute; a large drop would prove them
// on the critical path. Compile-time-elided at its default (ablate=false =>
// byte-identical to shipped). Gated by env SDPA_ABLATE_WRITER in the descriptor.
template <uint32_t cb, bool batch, bool ablate, typename Acc, typename PageFn>
FORCE_INLINE void write_tiles(uint32_t n, uint32_t tile_bytes, const Acc& acc, PageFn page_of) {
    if constexpr (batch) {
        cb_wait_front(cb, n);
        if constexpr (!ablate) {
            uint32_t rptr = get_read_ptr(cb);
            for (uint32_t t = 0; t < n; ++t) {
                noc_async_write_tile(page_of(t), acc, rptr);
                rptr += tile_bytes;
            }
            noc_async_write_barrier();  // ONE barrier for n writes -> up to n writes in flight
        }
        cb_pop_front(cb, n);
    } else {
        for (uint32_t t = 0; t < n; ++t) {
            cb_wait_front(cb, 1);
            if constexpr (!ablate) {
                noc_async_write_tile(page_of(t), acc, get_read_ptr(cb));
                noc_async_write_barrier();
            }
            cb_pop_front(cb, 1);
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(5);
    // R1b: the last q-chunk may be a partial block — the compute kernel packs only
    // sq_valid tile-rows to cb_out, so the writer drains exactly that many.

    // MEASUREMENT-ONLY writer NoC stub (see write_tiles). 0 for every shipped build.
    constexpr uint32_t ablate_writer_v = get_compile_time_arg_val(6);
    constexpr bool ablate_writer = ablate_writer_v != 0;

    constexpr auto dst_args = TensorAccessorArgs<7>();

    // R3 DM-batching knob (writer twin) — RE-MEASURED in R3a, kept PARKED at the
    // per-tile default to match the reader (see the reader kernel for the full
    // rationale: re-measured flat, reads hidden / compute-bound, parked to stay
    // byte-identical to the gate-passing R2/R3b writer and to avoid widening the
    // dormant zero-win batching under R3a's prefer-divisor chunking). The write_tiles
    // scaffolding stays a live tunable; re-enable with the divisor predicate
    // batch_q = (Sq_t % Sq_chunk_t) == 0 when the writes reach the critical path.
    constexpr bool batch_q = false;

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

        const uint32_t sq_off = qc * Sq_chunk_t;
        const uint32_t sq_valid = (Sq_chunk_t < Sq_t - sq_off) ? Sq_chunk_t : (Sq_t - sq_off);

        // Output block: (sq_valid x Dt) tiles, row-major (sq, d) — matches the
        // compute kernel's phase-11 pack order.
        const uint32_t base = (b * H + h) * Sq_t;
        write_tiles<cb_out, batch_q, ablate_writer>(sq_valid * Dt, tile_bytes, acc, [&](uint32_t t) {
            const uint32_t sq_g = sq_off + (t / Dt);
            const uint32_t d = t % Dt;
            return (base + sq_g) * Dt + d;
        });
    }
}
