// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// routed_expert_ffn reader: runs as T threads, one per compute thread. dfb::x and dfb::w have one tile counter per
// thread, so reader thread t feeds only Tensix t, in the exact order it consumes tiles. With implicit_sync (Quasar,
// chosen by the host) each read is tagged with a DFB transaction id and the DM0 ISR posts the credits once the reads
// land; otherwise the kernel does the explicit reserve_back/barrier/push_back sequence. For each tile row m
// that Tensix t owns (t, t + T, ...):
//   x[m, 0..Kt)                                  -> dfb::x (held by compute for all of phase 1)
//   for h: w_gate[0..Kt, h], then w_up[0..Kt, h] -> dfb::w
//   for n: w_down[0..Ht, n]                      -> dfb::w
// All matrices are row-major in tiles, so tile (r, c) of an R x C tile matrix is page r * C + c.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t Ht = get_arg(args::Ht);
    [[maybe_unused]] constexpr bool implicit_sync = get_arg(args::implicit_sync) != 0;

    Noc noc;
    DataflowBuffer dfb_x(dfb::x);
    DataflowBuffer dfb_w(dfb::w);

    const auto x = TensorAccessor(tensor::x);
    const auto w_gate = TensorAccessor(tensor::w_gate);
    const auto w_up = TensorAccessor(tensor::w_up);
    const auto w_down = TensorAccessor(tensor::w_down);

    auto push_tile = [&](DataflowBuffer& dfb, const auto& src, uint32_t page) {
#ifdef ARCH_QUASAR
        if constexpr (implicit_sync) {
            noc.async_read<NocOptions::TXN_ID>(src, dfb, {.page_id = page}, {});
            return;
        }
#endif
        dfb.reserve_back(1);
        noc.async_read(src, dfb, dfb.get_entry_size(), {.page_id = page}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    };

    for (uint32_t m = get_my_thread_id(); m < Mt; m += get_num_threads()) {
        for (uint32_t k = 0; k < Kt; ++k) {
            push_tile(dfb_x, x, m * Kt + k);
        }
        for (uint32_t h = 0; h < Ht; ++h) {
            for (uint32_t k = 0; k < Kt; ++k) {
                push_tile(dfb_w, w_gate, k * Ht + h);
            }
            for (uint32_t k = 0; k < Kt; ++k) {
                push_tile(dfb_w, w_up, k * Ht + h);
            }
        }
        for (uint32_t n = 0; n < Kt; ++n) {
            for (uint32_t h = 0; h < Ht; ++h) {
                push_tile(dfb_w, w_down, h * Kt + n);
            }
        }
    }

    // With implicit sync, posts any partial transaction-id batch of both DFBs before either destructor waits for
    // compute: compute needs the tail tiles of both to finish.
    dfb_x.finish();
    dfb_w.finish();
}
