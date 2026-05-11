// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for multigammaln_lanczos. Drains `num_tiles` tiles from
// cb_output_tiles and writes each one back to DRAM at the matching logical
// tile id.
//
// Runs on the data-movement RISC paired with NoC1 (BRISC per the default
// WriterConfigDescriptor binding on Wormhole/Blackhole).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_output_tiles = 16;

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    const uint32_t end_tile_id = start_tile_id + num_tiles;
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        cb_wait_front(cb_output_tiles, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        noc_async_write_tile(tile_id, output_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, 1);
    }
}
