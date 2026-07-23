// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t key_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t value_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t state_tiles = key_tiles * value_tiles;

    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t q_address = get_arg_val<uint32_t>(1);
    const uint32_t k_address = get_arg_val<uint32_t>(2);
    const uint32_t v_address = get_arg_val<uint32_t>(3);
    const uint32_t decay_address = get_arg_val<uint32_t>(4);
    const uint32_t beta_address = get_arg_val<uint32_t>(5);
    const uint32_t state_address = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_decay = 3;
    constexpr uint32_t cb_beta = 4;
    constexpr uint32_t cb_state = 5;
    constexpr uint32_t tile_size = get_tile_size(cb_state);

    constexpr auto q_args = TensorAccessorArgs<2>();
    const auto q_accessor = TensorAccessor(q_args, q_address, tile_size);
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto k_accessor = TensorAccessor(k_args, k_address, tile_size);
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto v_accessor = TensorAccessor(v_args, v_address, tile_size);
    constexpr auto decay_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    const auto decay_accessor = TensorAccessor(decay_args, decay_address, tile_size);
    constexpr auto beta_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();
    const auto beta_accessor = TensorAccessor(beta_args, beta_address, tile_size);
    constexpr auto state_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    const auto state_accessor = TensorAccessor(state_args, state_address, tile_size);

    Noc noc;
    CircularBuffer q_buffer(cb_q);
    CircularBuffer k_buffer(cb_k);
    CircularBuffer v_buffer(cb_v);
    CircularBuffer decay_buffer(cb_decay);
    CircularBuffer beta_buffer(cb_beta);
    CircularBuffer state_buffer(cb_state);

    q_buffer.reserve_back(key_tiles);
    k_buffer.reserve_back(key_tiles);
    decay_buffer.reserve_back(key_tiles);
    const uint32_t key_offset = head * key_tiles;
    for (uint32_t tile = 0; tile < key_tiles; ++tile) {
        noc.async_read(
            q_accessor, q_buffer, tile_size, {.page_id = key_offset + tile}, {.offset_bytes = tile * tile_size});
        noc.async_read(
            k_accessor, k_buffer, tile_size, {.page_id = key_offset + tile}, {.offset_bytes = tile * tile_size});
        noc.async_read(
            decay_accessor,
            decay_buffer,
            tile_size,
            {.page_id = key_offset + tile},
            {.offset_bytes = tile * tile_size});
    }
    noc.async_read_barrier();
    q_buffer.push_back(key_tiles);
    k_buffer.push_back(key_tiles);
    decay_buffer.push_back(key_tiles);

    v_buffer.reserve_back(value_tiles);
    const uint32_t value_offset = head * value_tiles;
    for (uint32_t tile = 0; tile < value_tiles; ++tile) {
        noc.async_read(
            v_accessor, v_buffer, tile_size, {.page_id = value_offset + tile}, {.offset_bytes = tile * tile_size});
    }
    noc.async_read_barrier();
    v_buffer.push_back(value_tiles);

    beta_buffer.reserve_back(1);
    noc.async_read(beta_accessor, beta_buffer, tile_size, {.page_id = head}, {.offset_bytes = 0});
    noc.async_read_barrier();
    beta_buffer.push_back(1);

    state_buffer.reserve_back(state_tiles);
    const uint32_t state_offset = head * state_tiles;
    for (uint32_t tile = 0; tile < state_tiles; ++tile) {
        noc.async_read(
            state_accessor,
            state_buffer,
            tile_size,
            {.page_id = state_offset + tile},
            {.offset_bytes = tile * tile_size});
    }
    noc.async_read_barrier();
    state_buffer.push_back(state_tiles);
}
