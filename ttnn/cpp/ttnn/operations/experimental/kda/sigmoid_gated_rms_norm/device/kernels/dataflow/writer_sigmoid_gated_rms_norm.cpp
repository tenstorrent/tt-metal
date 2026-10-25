// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/norm_items.hpp"

// Reads each group's gate, then writes the group's outputs. The gate and the inputs (the reader's) are the op's input
// traffic; carried on both NoCs, they land in half the time. The gate tensor holds heads [Ha, H) from
// gate_offset_tiles (the reader reads the other heads' activated gate).
template <
    uint32_t Vt,
    uint32_t H,
    uint32_t Ha,
    uint32_t Mt,
    uint32_t gate_row_tiles,
    uint32_t gate_offset_tiles,
    uint32_t G,
    uint32_t max_items>
TT_KERNEL void writer(uint32_t first_item, uint32_t item_stride, uint32_t item_count) {
    const auto g_acc = TensorAccessor(tensor::gate);
    const auto out_acc = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer out(dfb::out);
    const NormItems<H, Ha, Mt, max_items> core_items(first_item, item_stride, item_count);
    const uint32_t wi_count = item_count;

    // Item k's gate tiles land at `slot` items from the gate buffer's write pointer.
    auto read_gate = [&](uint32_t k, uint32_t slot) {
        const auto item = core_items[k];
        const uint32_t gate_base = (item.b * Mt + item.mt) * gate_row_tiles + gate_offset_tiles + (item.h - Ha) * Vt;
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
        const uint32_t raw = core_items.raw_in(first, items);
        if (raw > 0) {
            // The compute kernel starts on the first item's gate sigmoid: that gate lands on its own.
            gate.reserve_back(raw * Vt);
            read_gate(first, 0);
            noc.async_read_barrier();
            gate.push_back(Vt);
            if (raw > 1) {
                // Offsets from the write pointer, which the first gate's push moved on by one item.
                for (uint32_t j = 1; j < raw; j++) {
                    read_gate(first + j, j - 1);
                }
                noc.async_read_barrier();
                gate.push_back((raw - 1) * Vt);
            }
        }

        for (uint32_t j = 0; j < items; j++) {
            const auto item = core_items[first + j];
            const uint32_t out_base = (item.b * Mt + item.mt) * H * Vt + item.h * Vt;
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
