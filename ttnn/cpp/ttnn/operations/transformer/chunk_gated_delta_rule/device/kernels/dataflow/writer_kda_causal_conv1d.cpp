// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Qt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t worker_Ct = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr auto q_a = TensorAccessorArgs<5>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    const uint32_t mt_start = get_arg_val<uint32_t>(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(5);
    const auto q = TensorAccessor(q_a, q_addr, tile_bytes);
    const auto k = TensorAccessor(k_a, k_addr, tile_bytes);
    const auto v = TensorAccessor(v_a, v_addr, tile_bytes);
    Noc noc;
    CircularBuffer out(5);
    const uint32_t channel_tile_offset = (mt_start / Mt) * worker_Ct;

    for (uint32_t item = 0; item < mt_count; ++item) {
        const uint32_t work_item = mt_start + item;
        const uint32_t mt = work_item % Mt;
        out.wait_front(worker_Ct);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(out);
        for (uint32_t ct = 0; ct < worker_Ct; ++ct) {
            const uint32_t global_ct = channel_tile_offset + ct;
            if (global_ct < Qt) {
                noc.async_write(
                    src, q, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = mt * Qt + global_ct});
            } else if (global_ct < Qt + Kt) {
                const uint32_t kt = global_ct - Qt;
                noc.async_write(src, k, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = mt * Kt + kt});
            } else {
                const uint32_t vt = global_ct - Qt - Kt;
                noc.async_write(src, v, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = mt * Vt + vt});
            }
        }
        noc.async_write_barrier();
        out.pop_front(worker_Ct);
    }
}
