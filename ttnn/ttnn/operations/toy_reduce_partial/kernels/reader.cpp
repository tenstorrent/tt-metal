// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified reader for toy_reduce_partial.
//
// Streams input tiles only. The reduce scaler (including the partial-tile
// scaler for non-tile-aligned reduce dimensions) is owned by the compute
// kernel via ReduceScaler::compute_managed().

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_in = 0;

    // Stream input tiles
    uint32_t tile_bytes = get_tile_size(cb_in);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);
        noc_async_read_tile(i, accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
