// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused sigmoid-gated RMSNorm compute kernel. Besides writing the output it reads the gate: the
// reader then carries only x on its NOC, and the gate reads share the writer's NOC with the output writes, which
// balances the two NOCs (x 16 KB, gate + out 8 + 8 KB per unit at fp32 x / bf16 gate and output).
// The gate DFB holds `gate_depth` units; the writer keeps it full, reading unit i + gate_depth after unit i's output.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// gate_row_tiles: tiles per gate tile-row (gate padded width / 32; H*Vt for a [B,T,H*V] gate).
// gate_col_offset: first gate tile column read (0 unless the gate is a column window of a wider tensor).
template <uint32_t Vt, uint32_t H, uint32_t Mt, uint32_t gate_row_tiles, uint32_t gate_col_offset, uint32_t gate_depth>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto out_acc = TensorAccessor(tensor::output);
    const auto g_acc = TensorAccessor(tensor::gate);
    Noc noc;
    DataflowBuffer out(dfb::out);
    DataflowBuffer gate(dfb::gate);

    auto read_gate = [&](uint32_t i) {
        const uint32_t wi = wi_start + i;
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t gate_base = (b * Mt + mt) * gate_row_tiles + gate_col_offset + h * Vt;
        gate.reserve_back(Vt);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_read(
                g_acc,
                gate,
                gate.get_entry_size(),
                {.page_id = gate_base + vt},
                {.offset_bytes = vt * gate.get_entry_size()});
        }
        noc.async_read_barrier();
        gate.push_back(Vt);
    };

    const uint32_t prefetch = wi_count < gate_depth ? wi_count : gate_depth;
    for (uint32_t i = 0; i < prefetch; i++) {
        read_gate(i);
    }
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
        // The compute kernel pops gate(i) with its last output group, so a gate slot frees up about now.
        if (i + gate_depth < wi_count) {
            read_gate(i + gate_depth);
        }
        noc.async_write_barrier();
        out.pop_front(Vt);
    }
}
