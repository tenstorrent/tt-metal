// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Copy tile rows [row_begin, row_end) of `columns` tiles: source columns from source_column, destination from 0.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr auto source_args = TensorAccessorArgs<2>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const uint32_t source_addr = get_arg_val<uint32_t>(0);
    const uint32_t destination_addr = get_arg_val<uint32_t>(1);
    const uint32_t source_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t source_column = get_arg_val<uint32_t>(3);
    const uint32_t destination_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t row_begin = get_arg_val<uint32_t>(5);
    const uint32_t row_end = get_arg_val<uint32_t>(6);

    const uint32_t tile_bytes = get_tile_size(cb);
    const auto source = TensorAccessor(source_args, source_addr, tile_bytes);
    const auto destination = TensorAccessor(destination_args, destination_addr, tile_bytes);
    cb_reserve_back(cb, columns);
    const uint32_t buffer = get_write_ptr(cb);
    for (uint32_t row = row_begin; row < row_end; ++row) {
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_read_page(row * source_row_tiles + source_column + c, source, buffer + c * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_write_page(row * destination_row_tiles + c, destination, buffer + c * tile_bytes);
        }
        noc_async_write_barrier();
    }
}
