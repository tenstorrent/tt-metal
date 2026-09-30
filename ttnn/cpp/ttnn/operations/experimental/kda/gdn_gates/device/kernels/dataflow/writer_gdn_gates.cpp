// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

TT_KERNEL void writer(uint32_t mt_start, uint32_t mt_count) {
    const auto beta_acc = TensorAccessor(tensor::beta_out);
    const auto g_acc = TensorAccessor(tensor::g_out);
    Noc noc;
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer g(dfb::g);
    for (uint32_t i = 0; i < mt_count; i++) {
        const uint32_t mt = mt_start + i;
        beta.wait_front(1);
        g.wait_front(1);
        noc.async_write(beta, beta_acc, beta.get_entry_size(), {.offset_bytes = 0}, {.page_id = mt});
        noc.async_write(g, g_acc, g.get_entry_size(), {.offset_bytes = 0}, {.page_id = mt});
        noc.async_write_barrier();
        beta.pop_front(1);
        g.pop_front(1);
    }
}
