// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for multigammaln. Streams `num_tiles` consecutive input tiles starting
// at `start_tile_id` from DRAM into cb_input_tiles, one tile per push.
//
// Runs on the data-movement RISC paired with NoC0 (NCRISC per the default
// ReaderConfigDescriptor binding on Wormhole/Blackhole).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_input_tiles = 0;

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    const uint32_t end_tile_id = start_tile_id + num_tiles;
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        cb_reserve_back(cb_input_tiles, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        noc_async_read_tile(tile_id, input_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, 1);
    }
}
