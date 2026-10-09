// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// routed_expert_ffn writer: writes y to the DRAM-interleaved output. With implicit_sync (Quasar, chosen by the host)
// each write is tagged with a DFB transaction id and the DM0 ISR acks the credits once it is sent; otherwise the kernel
// does the explicit wait_front/barrier/pop_front sequence. Runs as W threads draining T
// compute threads through dfb::out, which has max(T, W) tile counters: counter c is filled by Tensix c % T and drained
// by writer c % W. Tensix t emits its rows t, t + T, ... one tile at a time, so its p-th tile is y[t + (p / Kt) * T, p
// % Kt].
//   W <= T: writer w owns Tensix w, w + W, ... and its pops rotate over them; pop i is Tensix w + (i % (T/W)) * W,
//           tile i / (T/W).
//   W > T:  writer w owns one counter of Tensix w % T, which gets that Tensix's tiles w / T, w / T + W/T, ...

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t T = get_arg(args::compute_threads);
    const uint32_t W = get_num_threads();
    const uint32_t w = get_my_thread_id();
    [[maybe_unused]] constexpr bool implicit_sync = get_arg(args::implicit_sync) != 0;

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t out_tile_bytes = dfb_out.get_entry_size();

    const auto y = TensorAccessor(tensor::y);

    auto write_tile = [&](uint32_t tensix, uint32_t p) {
        const uint32_t row = tensix + (p / Kt) * T;
#ifdef ARCH_QUASAR
        if constexpr (implicit_sync) {
            noc.async_write<NocOptions::TXN_ID>(dfb_out, y, {}, {.page_id = row * Kt + p % Kt});
            return;
        }
#endif
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, y, out_tile_bytes, {}, {.page_id = row * Kt + p % Kt});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    };

    const uint32_t tiles_per_tensix = (Mt / T) * Kt;
    if (W <= T) {
        const uint32_t tensix_per_writer = T / W;
        for (uint32_t i = 0; i < tensix_per_writer * tiles_per_tensix; ++i) {
            write_tile(w + (i % tensix_per_writer) * W, i / tensix_per_writer);
        }
    } else {
        const uint32_t writers_per_tensix = W / T;
        for (uint32_t p = w / T; p < tiles_per_tensix; p += writers_per_tensix) {
            write_tile(w % T, p);
        }
    }

    dfb_out.finish();
}
