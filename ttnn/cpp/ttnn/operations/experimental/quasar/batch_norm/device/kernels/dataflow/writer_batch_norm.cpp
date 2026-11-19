// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar batch_norm writer: W DM harts drain C NEOs through the STRIDED dfb::out ring.
//
// Ring geometry (host, C producers >= W consumers): consumer hart w owns entries w, w+W, ... and its k-th
// pop comes from producer NEO (w + (k mod C/W) * W). Hart w therefore drains NEOs w, w+W, ... in strict
// rotation, one tile each (the mirror of the reader's producer-side rotation).
//
// Output tiles arrive per plane in order 0..HtWt-1. Every NEO runs rounds = ceil(num_planes / C) rounds;
// NEO t's plane in round k is plane_start + k*C + t when real, else NEO 0's plane of that round (the
// compute recomputed it, so the bytes are identical to what NEO 0's drain stores).
//
// dfb::out uses implicit sync (noc.async_write<TXN_ID>, DM0 ISR acks credits back to the packer); the drain
// and write flush run in ~DataflowBuffer().
//
// Compile-time args: HtWt, compute_threads   Runtime args: plane_start, num_planes

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t plane_start = get_arg(args::plane_start);
    const uint32_t num_planes = get_arg(args::num_planes);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t C = get_arg(args::compute_threads);

    const uint32_t w = get_my_thread_id();
    const uint32_t W = get_num_threads();
    const uint32_t group = C / W;  // NEOs drained by this hart: w, w+W, ...
    const uint32_t rounds = (num_planes + C - 1) / C;

    Noc noc;
    const auto out_acc = TensorAccessor(tensor::output);
    DataflowBuffer out(dfb::out);

    // Plane of NEO t in round k: its own plane when real, else NEO 0's plane of that round.
    auto plane_of = [&](uint32_t t, uint32_t k) __attribute__((always_inline)) {
        const uint32_t local = k * C + t;
        return plane_start + (local < num_planes ? local : k * C);
    };

    for (uint32_t k = 0; k < rounds; ++k) {
        for (uint32_t t = 0; t < HtWt; ++t) {
            // One tile from each NEO of the group, in ring rotation order.
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_write<NocOptions::TXN_ID>(out, out_acc, {}, {.page_id = plane_of(w + j * W, k) * HtWt + t});
            }
        }
    }

    // No finish() / write_barrier() calls: ~DataflowBuffer() drains the ring and flushes the TXN_ID-tagged
    // writes.
}
