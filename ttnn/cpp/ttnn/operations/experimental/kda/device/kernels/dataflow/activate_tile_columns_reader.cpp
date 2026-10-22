// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Reads tiles [begin, end) of the rows x columns block (row-major) whose columns start at source_column, `group` at a
// time, for activate_tile_columns.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr auto source_args = TensorAccessorArgs<3>();
    const uint32_t source_addr = get_arg_val<uint32_t>(0);
    const uint32_t source_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t source_column = get_arg_val<uint32_t>(2);
    const uint32_t begin = get_arg_val<uint32_t>(3);
    const uint32_t end = get_arg_val<uint32_t>(4);

    const uint32_t tile_bytes = get_tile_size(cb);
    const auto source = TensorAccessor(source_args, source_addr, tile_bytes);
    for (uint32_t first = begin; first < end; first += group) {
        const uint32_t count = end - first < group ? end - first : group;
        cb_reserve_back(cb, count);
        const uint32_t buffer = get_write_ptr(cb);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = first + i;
            noc_async_read_page(
                (tile / columns) * source_row_tiles + source_column + tile % columns, source, buffer + i * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, count);
    }
}
