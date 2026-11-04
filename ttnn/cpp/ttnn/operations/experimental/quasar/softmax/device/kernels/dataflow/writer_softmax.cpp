// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar softmax writer: W DM harts drain C NEOs through the STRIDED dfb::out ring.
//
// Ring geometry (host, C producers >= W consumers): consumer hart w owns entries w, w+W, ... and its k-th
// pop comes from producer NEO (w + (k mod C/W) * W). Hart w therefore drains NEOs w, w+W, ... in strict
// rotation, one tile each (the mirror of the reader's producer-side rotation).
//
// Output tiles arrive per row in order 0..Wt-1. Every NEO runs rounds = ceil(NCHt / C) rows; NEO t's row
// in round k is row_start + k*C + t when real, else NEO 0's row of that round (the compute recomputed it,
// so the bytes are identical to what NEO 0's drain stores).
//
// dfb::out uses implicit sync (noc.async_write<TXN_ID>, DM0 ISR acks credits back to the packer); the drain
// and write flush run in ~DataflowBuffer().
//
// Compile-time args: Wt, compute_threads   Runtime args: row_start, NCHt

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t row_start = get_arg(args::row_start);
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t C = get_arg(args::compute_threads);

    const uint32_t w = get_my_thread_id();
    const uint32_t W = get_num_threads();
    const uint32_t group = C / W;  // NEOs drained by this hart: w, w+W, ...
    const uint32_t rounds = (NCHt + C - 1) / C;

    Noc noc;
    const auto out_acc = TensorAccessor(tensor::output);
    DataflowBuffer out(dfb::out);

    // Row of NEO t in round k: its own row when real, else NEO 0's row of that round.
    auto row_of = [&](uint32_t t, uint32_t k) __attribute__((always_inline)) {
        const uint32_t local = k * C + t;
        return row_start + (local < NCHt ? local : k * C);
    };

    for (uint32_t k = 0; k < rounds; ++k) {
        for (uint32_t q = 0; q < Wt; ++q) {
            // One tile from each NEO of the group, in ring rotation order.
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_write<NocOptions::TXN_ID>(out, out_acc, {}, {.page_id = row_of(w + j * W, k) * Wt + q});
            }
        }
    }

    // No finish() / write_barrier() calls: ~DataflowBuffer() drains the ring and flushes the TXN_ID-tagged
    // writes.
}
