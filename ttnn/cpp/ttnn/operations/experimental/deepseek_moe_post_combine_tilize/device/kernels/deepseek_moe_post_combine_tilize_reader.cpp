// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t cb_tilize_input_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t bytes_to_read_per_row = get_named_compile_time_arg_val("bytes_to_read_per_row");

    uint32_t rt_args_idx = 0;
    const uint32_t intra_row_byte_offset = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t row_page_offset = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr auto input_tensor_accessor_args = TensorAccessorArgs<0>();
    const auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_address);

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    CircularBuffer cb_tilize_input(cb_tilize_input_id);

    cb_tilize_input.reserve_back(tile_height);
    uint32_t l1_write_addr = cb_tilize_input.get_write_ptr();

    uint32_t page_id = row_page_offset;
    for (uint32_t row = 0; row < tile_height; ++row) {
        noc.async_read(
            input_tensor_accessor,
            CoreLocalMem<uint32_t>(l1_write_addr),
            bytes_to_read_per_row,
            {.page_id = page_id, .offset_bytes = intra_row_byte_offset},
            {});

        l1_write_addr += bytes_to_read_per_row;
        page_id++;
    }

    noc.async_read_barrier();
    cb_tilize_input.push_back(tile_height);
}
