// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
template <typename Accessor>
FORCE_INLINE void write_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_write(dfb, acc, entry, {.offset_bytes = t * entry}, {.page_id = first_page + t});
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}
}  // namespace

template <uint32_t Kt, uint32_t Vt>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto state_acc = TensorAccessor(tensor::state_out);
    const auto out_acc = TensorAccessor(tensor::out);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t h = wi_start + i;
        write_tiles(state_acc, hnew, noc, h * KV, KV);  // in-place state update
        write_tiles(out_acc, out, noc, h * Vt, Vt);
    }
}
