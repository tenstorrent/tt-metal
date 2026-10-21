// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar layer_norm (Welford) reader: R DM harts feed C NEOs through STRIDED rings.
//
// Ring geometry (host): dfb::in / gamma / beta have R producers and C consumers, so producer hart p owns
// entries p, p+R, ... and its k-th push lands with consumer (p + (k mod C/R) * R). Hart p therefore feeds
// NEOs p, p+R, p+2R, ... in strict rotation, one tile each, and keeps those NEOs' streams in lock-step.
// Every NEO runs rounds = ceil(NCHt / C) rows; NEO t's round-k row is row_start + k*C + t when that is a
// real row of this core, else NEO 0's row of the same round (a duplicate the writer stores again).
//
// Per row, in the order the compute consumes:
//   pass 1:  in <- tiles 0..Wt-1
//   pass 2:  per block of block_size tiles in processing order p: in <- src(p) for p >= tail (reload of
//            the head), then gamma <- src(p), then beta <- src(p)
//   where tail = min(Wt, cb_length/2), src(p) = (Wt - tail + p) mod Wt.
//
// in / gamma / beta use DFB implicit sync (noc.async_read<TXN_ID>, DM0 ISR posts credits in half-window
// batches); eps is filled by the RISC and pushed explicitly, so the host opts it out of implicit sync.
//
// Compile-time args: Wt, cb_length, block_size, compute_threads   Runtime args: row_start, NCHt, eps
// Defines: FUSE_GAMMA, FUSE_BETA

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

void kernel_main() {
    const uint32_t row_start = get_arg(args::row_start);
    const uint32_t NCHt = get_arg(args::NCHt);
    const uint32_t eps_bits = get_arg(args::eps);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t cb_length = get_arg(args::cb_length);
    constexpr uint32_t blk = get_arg(args::block_size);
    constexpr uint32_t C = get_arg(args::compute_threads);

    constexpr uint32_t tail = Wt < cb_length / 2 ? Wt : cb_length / 2;
    constexpr uint32_t head = Wt - tail;

    const uint32_t p = get_my_thread_id();
    const uint32_t R = get_num_threads();
    const uint32_t group = C / R;  // NEOs fed by this hart: p, p+R, ...
    const uint32_t rounds = (NCHt + C - 1) / C;

    Noc noc;
    const auto in_acc = TensorAccessor(tensor::input);
    DataflowBuffer in(dfb::in);
    DataflowBuffer eps(dfb::eps);
#ifdef FUSE_GAMMA
    const auto gamma_acc = TensorAccessor(tensor::gamma);
    DataflowBuffer gamma(dfb::gamma);
#endif
#ifdef FUSE_BETA
    const auto beta_acc = TensorAccessor(tensor::beta);
    DataflowBuffer beta(dfb::beta);
#endif

    // Row of NEO t in round k: its own row when real, else NEO 0's row of that round.
    auto row_of = [&](uint32_t t, uint32_t k) __attribute__((always_inline)) {
        const uint32_t local = k * C + t;
        return row_start + (local < NCHt ? local : k * C);
    };
    auto src_of = [](uint32_t q) __attribute__((always_inline)) { return q < tail ? head + q : q - tail; };

    // One eps column-scalar tile per NEO in this hart's group (explicit sync, pushed in rotation order).
    for (uint32_t j = 0; j < group; ++j) {
        generate_bcast_col_scalar(eps, eps_bits);
    }

    for (uint32_t k = 0; k < rounds; ++k) {
        // pass 1: the whole row, rotating over the group per tile
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_read<NocOptions::TXN_ID>(in_acc, in, {.page_id = row_of(p + j * R, k) * Wt + w}, {});
            }
        }
        // pass 2: per block, reload the head tiles, then gamma, then beta
        for (uint32_t bp = 0; bp < Wt; bp += blk) {
            const uint32_t nb = (Wt - bp) < blk ? (Wt - bp) : blk;
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t q = bp + i;
                if (q >= tail) {
                    for (uint32_t j = 0; j < group; ++j) {
                        noc.async_read<NocOptions::TXN_ID>(
                            in_acc, in, {.page_id = row_of(p + j * R, k) * Wt + src_of(q)}, {});
                    }
                }
            }
#ifdef FUSE_GAMMA
            for (uint32_t i = 0; i < nb; ++i) {
                for (uint32_t j = 0; j < group; ++j) {
                    noc.async_read<NocOptions::TXN_ID>(gamma_acc, gamma, {.page_id = src_of(bp + i)}, {});
                }
            }
#endif
#ifdef FUSE_BETA
            for (uint32_t i = 0; i < nb; ++i) {
                for (uint32_t j = 0; j < group; ++j) {
                    noc.async_read<NocOptions::TXN_ID>(beta_acc, beta, {.page_id = src_of(bp + i)}, {});
                }
            }
#endif
        }
    }

    // No finish() calls: each ~DataflowBuffer() (reverse declaration order) tops up a partial ISR batch
    // and drains the ring. The host sizes the windows so no ring ever ends on a partial batch, so
    // the drains cannot deadlock on each other (see the program factory).
}
