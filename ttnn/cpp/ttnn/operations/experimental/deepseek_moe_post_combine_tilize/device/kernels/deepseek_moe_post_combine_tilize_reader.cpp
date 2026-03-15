// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    constexpr uint32_t tilize_input_cb_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t input_row_page_size = get_named_compile_time_arg_val("input_row_page_size");
    constexpr uint32_t bytes_to_read_per_row = get_named_compile_time_arg_val("bytes_to_read_per_row");

    uint32_t rt_args_idx = 0;
    const uint32_t intra_row_byte_offset = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t row_page_offset = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr auto input_tensor_accessor_args = TensorAccessorArgs<0>();
    const auto input_tensor_accessor =
        TensorAccessor(input_tensor_accessor_args, input_tensor_address, input_row_page_size);

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    cb_reserve_back(tilize_input_cb_id, tile_height);
    uint32_t l1_write_addr = get_write_ptr(tilize_input_cb_id);

    uint32_t page_id = row_page_offset;
    for (uint32_t row = 0; row < tile_height; ++row) {
        uint64_t noc_addr = get_noc_addr(page_id, input_tensor_accessor, intra_row_byte_offset);
        noc_async_read(noc_addr, l1_write_addr, bytes_to_read_per_row);

        l1_write_addr += bytes_to_read_per_row;
        page_id++;
    }

    noc_async_read_barrier();
    cb_push_back(tilize_input_cb_id, tile_height);
}
