// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Vt, uint32_t H, uint32_t Mt>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto out_acc = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer out(dfb::out);
    for (uint32_t i = 0; i < wi_count; i++) {
        const uint32_t wi = wi_start + i;
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t out_base = (b * Mt + mt) * H * Vt + h * Vt;
        out.wait_front(Vt);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_write(
                out,
                out_acc,
                out.get_entry_size(),
                {.offset_bytes = vt * out.get_entry_size()},
                {.page_id = out_base + vt});
        }
        noc.async_write_barrier();
        out.pop_front(Vt);
    }
}
