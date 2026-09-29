// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused sigmoid-gated RMSNorm compute kernel (sigmoid_gated_rms_norm_fused.cpp). It reads x, the
// weight and the scaler; the gate is read by the fused writer (writer_sigmoid_gated_rms_norm_fused.cpp) on the
// other NOC. Differences from reader_sigmoid_gated_rms_norm.cpp:
//   - x of the core's even units goes to DFB x0, of the odd units to x1. The compute kernel reads unit u+1 while
//     unit u is still in use, and each unit then starts at the front of its own DFB (no ring wrap in the tile index).
//   - No epsilon tile (the compute kernel adds epsilon on the SFPU).
//   - x of units 0 and 1 is pushed without waiting for the weight, and only rows 0-1 of each weight tile are read.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

template <uint32_t Vt>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto x_acc = TensorAccessor(tensor::input);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer x0(dfb::x0);
    DataflowBuffer x1(dfb::x1);
    DataflowBuffer weight(dfb::weight);
    Noc noc;

    // Reads are tagged: x with kXTrid, the weight with kWTrid, so that x of the first units is pushed without
    // waiting for the (slower, shared) weight reads.
    constexpr uint32_t kXTrid = 1;
    constexpr uint32_t kWTrid = 2;
    auto read_unit = [&](uint32_t i) {
        const uint32_t x_base = (wi_start + i) * Vt;
        DataflowBuffer& x = (i & 1) ? x1 : x0;
        x.reserve_back(Vt);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_read<NocOptions::TXN_ID>(
                x_acc,
                x,
                x.get_entry_size(),
                {.page_id = x_base + vt},
                {.offset_bytes = vt * x.get_entry_size()},
                {.trid = kXTrid});
        }
        return &x;
    };
    auto x_barrier = [&]() { noc.async_read_barrier<NocOptions::TXN_ID>({.trid = kXTrid}); };

    // The compute kernel uses only row 0 of each weight tile (row-broadcast copy): row 0 of face 0 (columns 0-15)
    // and of face 1 (columns 16-31). Every core reads the same weight pages, so read only up to row 1 of face 1:
    // one 576 B read (DRAM-aligned) per tile, starting at a different page on each core. The weight reads go first
    // (the compute kernel needs the weight at its first pass C, after passes A/B of unit 0 and pass A of unit 1).
    constexpr uint32_t kWeightReadBytes = 16 * 16 * 2 + 64;  // bf16 face 0 + rows 0-1 of face 1
    weight.reserve_back(Vt);
    const uint32_t w_rot = (wi_count > 0 ? wi_start / wi_count : 0) % Vt;
    for (uint32_t n = 0; n < Vt; n++) {
        const uint32_t vt = (n + w_rot) % Vt;
        noc.async_read<NocOptions::TXN_ID>(
            w_acc,
            weight,
            kWeightReadBytes,
            {.page_id = vt},
            {.offset_bytes = vt * weight.get_entry_size()},
            {.trid = kWTrid});
    }
    if (wi_count > 0) {
        DataflowBuffer* x_first = read_unit(0);
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            Vt * tt::constants::TILE_WIDTH>();
        x_barrier();
        x_first->push_back(Vt);
    }
    if (wi_count > 1) {
        DataflowBuffer* x_second = read_unit(1);
        x_barrier();
        x_second->push_back(Vt);
    }
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = kWTrid});
    weight.push_back(Vt);

    for (uint32_t i = 2; i < wi_count; i++) {
        DataflowBuffer* x = read_unit(i);
        x_barrier();
        x->push_back(Vt);
    }
    noc_async_read_set_trid(0, noc.get_noc_id());  // leave the read command buffer untagged
}
