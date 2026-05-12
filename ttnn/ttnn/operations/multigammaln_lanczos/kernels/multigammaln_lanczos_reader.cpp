// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multigammaln_lanczos — Reader kernel.
//
// Streams float32 tiles from DRAM into ``cb_input_tiles`` in tile-id order.
// One tile per loop iteration; reader↔compute streaming is double-buffered by
// the CB. The same tile is read four times by the compute kernel (one per
// Lanczos lgamma sub-evaluation) before being popped — that is handled on
// the compute side; the reader pushes each tile once.
//
// CT args: [cb_input_tiles_index, TensorAccessorArgs...]
// RT args: [src_addr, num_tiles, start_tile_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto src_accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    const uint32_t end_tile_id = start_tile_id + num_tiles;
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        cb_reserve_back(cb_input_tiles, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        noc_async_read_tile(tile_id, src_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, 1);
    }
}
