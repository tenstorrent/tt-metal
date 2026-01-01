// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar softmax reader: R DM harts feed C NEOs through the STRIDED dfb::in ring.
//
// Ring geometry (host): dfb::in has R producers and C consumers, so producer hart p owns entries p, p+R, ...
// and its k-th push lands with consumer (p + (k mod C/R) * R). Hart p therefore feeds NEOs p, p+R, ... in
// strict rotation, one tile each, and keeps those NEOs' streams in lock-step. Every NEO runs
// rounds = ceil(NCHt / C) rows; NEO t's round-k row is row_start + k*C + t when that is a real row of this
// core, else NEO 0's row of the same round (a duplicate the writer stores again).
//
// Per row: tiles 0..Wt-1, streamed once per compute pass (3 passes, 2 without NUMERIC_STABLE).
//
// dfb::in uses DFB implicit sync (noc.async_read<TXN_ID>, DM0 ISR posts credits in half-window batches);
// the reduce-scaler tile is filled by the RISC and pushed explicitly, so the host opts it out of implicit
// sync.
//
// Compile-time args: Wt, compute_threads   Runtime args: row_start, NCHt   Defines: NUMERIC_STABLE

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Reduce scaler for a 32x32 bf16 tile: row 0 of every 16x16 face = 1.0, everything else 0 (the layout the
// reduce LLK reads for REDUCE_ROW). The tile is 512 u32 words; a face is 128 words, its row 0 is 8 words.
template <typename DFB>
FORCE_INLINE void generate_reduce_scaler_one(DFB& dfb) {
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    for (uint32_t i = 0; i < 512; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t j = 0; j < 8; ++j) {
            ptr[face * 128 + j] = 0x3f803f80u;
        }
    }
    dfb.push_back(1);
}

void kernel_main() {
    const uint32_t row_start = get_arg(args::row_start);
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t C = get_arg(args::compute_threads);
#ifdef NUMERIC_STABLE
    constexpr uint32_t passes = 3;
#else
    constexpr uint32_t passes = 2;
#endif

    const uint32_t p = get_my_thread_id();
    const uint32_t R = get_num_threads();
    const uint32_t group = C / R;  // NEOs fed by this hart: p, p+R, ...
    const uint32_t rounds = (NCHt + C - 1) / C;

    Noc noc;
    const auto in_acc = TensorAccessor(tensor::input);
    DataflowBuffer in(dfb::in);
    DataflowBuffer scaler(dfb::scaler);

    // Row of NEO t in round k: its own row when real, else NEO 0's row of that round.
    auto row_of = [&](uint32_t t, uint32_t k) __attribute__((always_inline)) {
        const uint32_t local = k * C + t;
        return row_start + (local < NCHt ? local : k * C);
    };

    // One scaler tile per NEO in this hart's group (explicit sync, pushed in rotation order).
    for (uint32_t j = 0; j < group; ++j) {
        generate_reduce_scaler_one(scaler);
    }

    for (uint32_t k = 0; k < rounds; ++k) {
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t w = 0; w < Wt; ++w) {
                for (uint32_t j = 0; j < group; ++j) {
                    noc.async_read<NocOptions::TXN_ID>(in_acc, in, {.page_id = row_of(p + j * R, k) * Wt + w}, {});
                }
            }
        }
    }

    // No finish() calls: each ~DataflowBuffer() (reverse declaration order) tops up a partial ISR batch
    // and drains the ring. The host sizes the window so no ring ever ends on a partial batch.
}
