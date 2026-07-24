// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr auto g_a = TensorAccessorArgs<3>();
    constexpr auto w_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();

    const uint32_t consumer_index = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t consumer_count = get_arg_val<uint32_t>(4);
    const uint32_t producer_count = get_arg_val<uint32_t>(5);
    const uint32_t ready_semaphore_id = get_arg_val<uint32_t>(6);

    const uint32_t tg = get_tile_size(1);
    const uint32_t tw = get_tile_size(2);
    const auto g_acc = TensorAccessor(g_a, g_addr, tg);
    const auto w_acc = TensorAccessor(w_a, w_addr, tw);
    Noc noc;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<8, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    CircularBuffer weight(2);
    weight.reserve_back(Vt);
    for (uint32_t vt = 0; vt < Vt; vt++) {
        noc.async_read(w_acc, weight, tw, {.page_id = vt}, {.offset_bytes = vt * tw});
    }
    noc.async_read_barrier();
    weight.push_back(Vt);

    for (uint32_t i = 0; i < wi_count; i++) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_semaphore_id)),
            producer_count * (i + 1));
        const uint32_t wi = consumer_index + i * consumer_count;
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t gate_base = (b * Mt + mt) * H * Vt + h * Vt;
        CircularBuffer x(0);
        CircularBuffer gate(1);
        x.reserve_back(Vt);
        gate.reserve_back(Vt);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_read(g_acc, gate, tg, {.page_id = gate_base + vt}, {.offset_bytes = vt * tg});
        }
        noc.async_read_barrier();
        x.push_back(Vt);
        gate.push_back(Vt);
    }
}
