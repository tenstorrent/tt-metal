// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Writes the activated tiles [begin, end) of the rows x columns block (row-major) into destination's first columns.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr auto destination_args = TensorAccessorArgs<3>();
    const uint32_t destination_addr = get_arg_val<uint32_t>(0);
    const uint32_t destination_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t begin = get_arg_val<uint32_t>(2);
    const uint32_t end = get_arg_val<uint32_t>(3);

    const uint32_t tile_bytes = get_tile_size(cb);
    const auto destination = TensorAccessor(destination_args, destination_addr, tile_bytes);
    for (uint32_t first = begin; first < end; first += group) {
        const uint32_t count = end - first < group ? end - first : group;
        cb_wait_front(cb, count);
        const uint32_t buffer = get_read_ptr(cb);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = first + i;
            noc_async_write_page(
                (tile / columns) * destination_row_tiles + tile % columns, destination, buffer + i * tile_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, count);
    }
    noc_async_write_barrier();
}
