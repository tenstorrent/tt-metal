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
    const uint32_t output_address = get_arg_val<uint32_t>(1);
    const uint32_t state_address = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = 13;
    constexpr uint32_t cb_final_state = 14;
    constexpr uint32_t tile_size = get_tile_size(cb_output);

    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output_accessor = TensorAccessor(output_args, output_address, tile_size);
    constexpr auto state_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto state_accessor = TensorAccessor(state_args, state_address, tile_size);

    Noc noc;
    CircularBuffer output_buffer(cb_output);
    CircularBuffer state_buffer(cb_final_state);

    output_buffer.wait_front(value_tiles);
    const uint32_t output_offset = head * value_tiles;
    for (uint32_t tile = 0; tile < value_tiles; ++tile) {
        noc.async_write(
            output_buffer,
            output_accessor,
            tile_size,
            {.offset_bytes = tile * tile_size},
            {.page_id = output_offset + tile});
    }
    noc.async_write_barrier();
    output_buffer.pop_front(value_tiles);

    state_buffer.wait_front(state_tiles);
    const uint32_t state_offset = head * state_tiles;
    for (uint32_t tile = 0; tile < state_tiles; ++tile) {
        noc.async_write(
            state_buffer,
            state_accessor,
            tile_size,
            {.offset_bytes = tile * tile_size},
            {.page_id = state_offset + tile});
    }
    noc.async_write_barrier();
    state_buffer.pop_front(state_tiles);
}
