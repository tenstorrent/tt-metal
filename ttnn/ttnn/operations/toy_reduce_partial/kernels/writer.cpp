// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for toy_reduce_partial: writes reduced output tiles to DRAM.
// Shared by both REDUCE_ROW and REDUCE_COL modes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_out = 16;
    uint32_t tile_bytes = get_tile_size(cb_out);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
