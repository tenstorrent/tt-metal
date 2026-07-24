// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t channels = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr auto input_a = TensorAccessorArgs<3>();
    constexpr auto state_a = TensorAccessorArgs<input_a.next_compile_time_args_offset()>();
    constexpr auto tap0_a = TensorAccessorArgs<state_a.next_compile_time_args_offset()>();
    constexpr auto tap1_a = TensorAccessorArgs<tap0_a.next_compile_time_args_offset()>();
    constexpr auto tap2_a = TensorAccessorArgs<tap1_a.next_compile_time_args_offset()>();
    constexpr auto tap3_a = TensorAccessorArgs<tap2_a.next_compile_time_args_offset()>();
    static_assert(row_bytes == channels * sizeof(uint16_t));

    const uint32_t mt_start = get_arg_val<uint32_t>(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(1);
    const uint32_t input_addr = get_arg_val<uint32_t>(2);
    const uint32_t state_addr = get_arg_val<uint32_t>(3);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(4);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(5);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(6);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t tile_bytes = 2048;
    const auto input = TensorAccessor(input_a, input_addr, row_bytes);
    const auto state = TensorAccessor(state_a, state_addr, row_bytes);
    const auto tap0 = TensorAccessor(tap0_a, tap0_addr, tile_bytes);
    const auto tap1 = TensorAccessor(tap1_a, tap1_addr, tile_bytes);
    const auto tap2 = TensorAccessor(tap2_a, tap2_addr, tile_bytes);
    const auto tap3 = TensorAccessor(tap3_a, tap3_addr, tile_bytes);
    Noc noc;

    CircularBuffer weights(2);
    weights.reserve_back(4 * Ct);
    auto weight_dst = use<CircularBuffer::AddrSelector::WRITE_PTR>(weights);
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        noc.async_read(tap0, weight_dst, tile_bytes, {.page_id = ct}, {.offset_bytes = ct * tile_bytes});
        noc.async_read(tap1, weight_dst, tile_bytes, {.page_id = ct}, {.offset_bytes = (Ct + ct) * tile_bytes});
        noc.async_read(tap2, weight_dst, tile_bytes, {.page_id = ct}, {.offset_bytes = (2 * Ct + ct) * tile_bytes});
        noc.async_read(tap3, weight_dst, tile_bytes, {.page_id = ct}, {.offset_bytes = (3 * Ct + ct) * tile_bytes});
    }
    noc.async_read_barrier();
    weights.push_back(4 * Ct);

    CircularBuffer activation(0);
    for (uint32_t item = 0; item < mt_count; ++item) {
        const uint32_t mt = mt_start + item;
        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(Ct);
            auto activation_dst = use<CircularBuffer::AddrSelector::WRITE_PTR>(activation);
            for (uint32_t row = 0; row < 32; ++row) {
                const int32_t source_row = static_cast<int32_t>(mt * 32 + row + tap) - 3;
                if (source_row < 0) {
                    noc.async_read(
                        state,
                        activation_dst,
                        row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row + 3)},
                        {.offset_bytes = row * row_bytes});
                } else {
                    noc.async_read(
                        input,
                        activation_dst,
                        row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row)},
                        {.offset_bytes = row * row_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(Ct);
        }
    }
}
