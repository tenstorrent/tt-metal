// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// routed_expert_ffn reader: pushes tiles in the exact order compute consumes them, one thread, explicit sync.
// Each push_back moves to the next compute thread's tile counter, so push j of a DFB goes to thread j % T. Thread t
// owns tile rows t, t + T, ...; rows are handled in rounds of T, one per thread. For each round:
//   x[row of t, k] for k, then t                 -> dfb::x (held by compute for all of phase 1)
//   for h: w_gate[0..Kt, h], then w_up[0..Kt, h] -> dfb::w, each tile once per thread
//   for n: w_down[0..Ht, n]                      -> dfb::w, each tile once per thread
// All matrices are row-major in tiles, so tile (r, c) of an R x C tile matrix is page r * C + c.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t T = get_arg(args::compute_threads);

    Noc noc;
    DataflowBuffer dfb_x(dfb::x);
    DataflowBuffer dfb_w(dfb::w);

    const auto x = TensorAccessor(tensor::x);
    const auto w_gate = TensorAccessor(tensor::w_gate);
    const auto w_up = TensorAccessor(tensor::w_up);
    const auto w_down = TensorAccessor(tensor::w_down);

    auto push_tile = [&](DataflowBuffer& dfb, const auto& src, uint32_t page) {
        dfb.reserve_back(1);
        noc.async_read(src, dfb, dfb.get_entry_size(), {.page_id = page}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    };

    auto push_to_all = [&](const auto& src, uint32_t page) {
        for (uint32_t t = 0; t < T; ++t) {
            push_tile(dfb_w, src, page);
        }
    };

    for (uint32_t round_start = 0; round_start < Mt; round_start += T) {
        for (uint32_t k = 0; k < Kt; ++k) {
            for (uint32_t t = 0; t < T; ++t) {
                push_tile(dfb_x, x, (round_start + t) * Kt + k);
            }
        }
        for (uint32_t h = 0; h < Ht; ++h) {
            for (uint32_t k = 0; k < Kt; ++k) {
                push_to_all(w_gate, k * Ht + h);
            }
            for (uint32_t k = 0; k < Kt; ++k) {
                push_to_all(w_up, k * Ht + h);
            }
        }
        for (uint32_t n = 0; n < Kt; ++n) {
            for (uint32_t h = 0; h < Ht; ++h) {
                push_to_all(w_down, h * Kt + n);
            }
        }
    }

    dfb_x.finish();
    dfb_w.finish();
}
