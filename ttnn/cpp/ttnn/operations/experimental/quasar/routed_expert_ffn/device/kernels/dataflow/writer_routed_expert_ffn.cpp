// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// routed_expert_ffn writer: writes y to the DRAM-interleaved output, one thread, explicit sync. Each pop_front moves to
// the next compute thread's tile counter, so pop j comes from thread j % T. Thread t emits its tile rows t, t + T, ...
// one tile at a time, so for each round of T rows the writer takes column n of every thread's row in turn.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t T = get_arg(args::compute_threads);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t out_tile_bytes = dfb_out.get_entry_size();

    const auto y = TensorAccessor(tensor::y);

    for (uint32_t round_start = 0; round_start < Mt; round_start += T) {
        for (uint32_t n = 0; n < Kt; ++n) {
            for (uint32_t t = 0; t < T; ++t) {
                dfb_out.wait_front(1);
                noc.async_write(dfb_out, y, out_tile_bytes, {}, {.page_id = (round_start + t) * Kt + n});
                noc.async_write_barrier();
                dfb_out.pop_front(1);
            }
        }
    }

    dfb_out.finish();
}
