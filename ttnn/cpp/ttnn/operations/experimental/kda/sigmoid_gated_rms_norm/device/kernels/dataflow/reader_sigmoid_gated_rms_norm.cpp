// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Vt, uint32_t H, uint32_t Mt, uint32_t gate_row_tiles, uint32_t gate_offset_tiles>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto x_acc = TensorAccessor(tensor::input);
    const auto g_acc = TensorAccessor(tensor::gate);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer ones(dfb::ones);
    Noc noc;

    // A BF16 tile of ones: the exact right operand of the compute kernel's row-sum matmul.
    ones.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ones_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());
    for (uint32_t j = 0; j < ones.get_entry_size() / sizeof(uint32_t); j++) {
        ones_ptr[j] = 0x3F803F80;
    }
    ones.push_back(1);

    weight.reserve_back(Vt);
    for (uint32_t vt = 0; vt < Vt; vt++) {
        noc.async_read(
            w_acc, weight, weight.get_entry_size(), {.page_id = vt}, {.offset_bytes = vt * weight.get_entry_size()});
    }
    noc.async_read_barrier();
    weight.push_back(Vt);

    for (uint32_t i = 0; i < wi_count; i++) {
        const uint32_t wi = wi_start + i;
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t x_base = wi * Vt;
        const uint32_t gate_base = (b * Mt + mt) * gate_row_tiles + gate_offset_tiles + h * Vt;
        x.reserve_back(Vt);
        gate.reserve_back(Vt);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_read(
                x_acc, x, x.get_entry_size(), {.page_id = x_base + vt}, {.offset_bytes = vt * x.get_entry_size()});
            noc.async_read(
                g_acc,
                gate,
                gate.get_entry_size(),
                {.page_id = gate_base + vt},
                {.offset_bytes = vt * gate.get_entry_size()});
        }
        noc.async_read_barrier();
        x.push_back(Vt);
        gate.push_back(Vt);
    }
}
