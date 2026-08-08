// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t block_ct = get_compile_time_arg_val(0);
    constexpr uint32_t channels = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr auto input_a = TensorAccessorArgs<4>();
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
    CircularBuffer activation(0);
    auto load_weights = [&](uint32_t ct_start) {
        weights.reserve_back(4 * block_ct);
        auto weight_dst = use<CircularBuffer::AddrSelector::WRITE_PTR>(weights);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            const uint32_t source_ct = ct_start + ct;
            noc.async_read(tap0, weight_dst, tile_bytes, {.page_id = source_ct}, {.offset_bytes = ct * tile_bytes});
            noc.async_read(
                tap1, weight_dst, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (block_ct + ct) * tile_bytes});
            noc.async_read(
                tap2,
                weight_dst,
                tile_bytes,
                {.page_id = source_ct},
                {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
            noc.async_read(
                tap3,
                weight_dst,
                tile_bytes,
                {.page_id = source_ct},
                {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
        }
        noc.async_read_barrier();
        weights.push_back(4 * block_ct);
    };
    if constexpr (num_blocks == 1) {
        load_weights(0);
    }
    constexpr uint32_t block_row_bytes = block_ct * 32 * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = 32 * sizeof(uint16_t);
    for (uint32_t item = 0; item < mt_count; ++item) {
        const uint32_t work = mt_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;

        if constexpr (num_blocks > 1) {
            load_weights(ct_start);
        }

        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(block_ct);
            auto activation_dst = use<CircularBuffer::AddrSelector::WRITE_PTR>(activation);
            for (uint32_t row = 0; row < 32; ++row) {
                const int32_t source_row = static_cast<int32_t>(mt * 32 + row + tap) - 3;
                if (source_row < 0) {
                    noc.async_read(
                        state,
                        activation_dst,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row + 3),
                         .offset_bytes = ct_start * block_offset_scale},
                        {.offset_bytes = row * block_row_bytes});
                } else {
                    noc.async_read(
                        input,
                        activation_dst,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row), .offset_bytes = ct_start * block_offset_scale},
                        {.offset_bytes = row * block_row_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(block_ct);
        }
    }
}
