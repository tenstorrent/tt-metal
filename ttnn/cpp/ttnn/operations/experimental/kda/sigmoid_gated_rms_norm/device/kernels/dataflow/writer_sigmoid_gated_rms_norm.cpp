// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Reads each group's gate, then writes the group's outputs. The gate and the inputs (the reader's) are the op's input
// traffic; carried on both NoCs, they land in half the time.
template <uint32_t Vt, uint32_t H, uint32_t Mt, uint32_t gate_row_tiles, uint32_t gate_offset_tiles, uint32_t G>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto g_acc = TensorAccessor(tensor::gate);
    const auto out_acc = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer out(dfb::out);

    // Item j's gate tiles land at `slot` items from the gate buffer's write pointer.
    auto read_gate = [&](uint32_t wi, uint32_t slot) {
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t gate_base = (bh / H * Mt + mt) * gate_row_tiles + gate_offset_tiles + bh % H * Vt;
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_read(
                g_acc,
                gate,
                gate.get_entry_size(),
                {.page_id = gate_base + vt},
                {.offset_bytes = (slot * Vt + vt) * gate.get_entry_size()});
        }
    };

    for (uint32_t first = 0; first < wi_count; first += G) {
        const uint32_t items = wi_count - first < G ? wi_count - first : G;
        // The compute kernel starts on the first item's gate sigmoid: that gate lands on its own.
        gate.reserve_back(items * Vt);
        read_gate(wi_start + first, 0);
        noc.async_read_barrier();
        gate.push_back(Vt);
        if (items > 1) {
            // Offsets from the write pointer, which the first gate's push moved on by one item.
            for (uint32_t j = 1; j < items; j++) {
                read_gate(wi_start + first + j, j - 1);
            }
            noc.async_read_barrier();
            gate.push_back((items - 1) * Vt);
        }

        for (uint32_t j = 0; j < items; j++) {
            const uint32_t wi = wi_start + first + j;
            const uint32_t bh = wi / Mt;
            const uint32_t mt = wi % Mt;
            const uint32_t out_base = (bh / H * Mt + mt) * H * Vt + bh % H * Vt;
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
}
