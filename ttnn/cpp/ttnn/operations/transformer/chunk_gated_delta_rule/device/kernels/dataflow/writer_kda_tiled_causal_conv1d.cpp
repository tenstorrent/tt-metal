// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Qt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t Ct = get_compile_time_arg_val(3);
    constexpr uint32_t worker_Ct = get_compile_time_arg_val(4);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t worker_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t Mt = get_compile_time_arg_val(7);
    constexpr auto q_a = TensorAccessorArgs<8>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto state_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(1);
    const uint32_t write_state = get_arg_val<uint32_t>(2);
    const uint32_t q_addr = get_arg_val<uint32_t>(3);
    const uint32_t k_addr = get_arg_val<uint32_t>(4);
    const uint32_t v_addr = get_arg_val<uint32_t>(5);
    const uint32_t state_addr = get_arg_val<uint32_t>(6);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t tile_width = 32;
    const uint32_t channel_block = work_start / Mt;
    const uint32_t mt_start = work_start % Mt;
    const uint32_t channel_tile_offset = channel_block * worker_Ct;
    const uint32_t channel_byte_offset = channel_tile_offset * tile_width * sizeof(uint16_t);
    const auto q = TensorAccessor(q_a, q_addr, tile_bytes);
    const auto k = TensorAccessor(k_a, k_addr, tile_bytes);
    const auto v = TensorAccessor(v_a, v_addr, tile_bytes);
    const auto state = TensorAccessor(state_a, state_addr, row_bytes);
    CircularBuffer output_cb(5);
    CircularBuffer state_cb(7);
    Noc noc;
    for (uint32_t item = 0; item < mt_count; ++item) {
        output_cb.wait_front(worker_Ct);
        const uint32_t mt = mt_start + item;
        for (uint32_t ct = 0; ct < worker_Ct; ++ct) {
            const uint32_t qkv_ct = channel_tile_offset + ct;
            if (qkv_ct < Qt) {
                noc.async_write(
                    output_cb, q, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = mt * Qt + qkv_ct});
            } else if (qkv_ct < Qt + Kt) {
                noc.async_write(
                    output_cb, k, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = mt * Kt + qkv_ct - Qt});
            } else {
                noc.async_write(
                    output_cb,
                    v,
                    tile_bytes,
                    {.offset_bytes = ct * tile_bytes},
                    {.page_id = mt * Vt + qkv_ct - Qt - Kt});
            }
        }
        noc.async_write_barrier();
        output_cb.pop_front(worker_Ct);
    }
    if (write_state) {
        state_cb.wait_front(3);
        for (uint32_t row = 0; row < 3; ++row) {
            noc.async_write(
                state_cb,
                state,
                worker_row_bytes,
                {.offset_bytes = row * worker_row_bytes},
                {.page_id = row, .offset_bytes = channel_byte_offset});
        }
        noc.async_write_barrier();
        state_cb.pop_front(3);
    }
}
