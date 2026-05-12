// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// glu_fused — Writer kernel.
//
// Drains cb_output_tiles one float32 tile at a time and writes back to DRAM at
// the corresponding output tile-id. Output tile-ids run from start_out_tile_id
// to start + num_per_core in row-major tile order; tile_id ↔ (n, c, h, w)
// mapping is handled by TensorAccessor.
//
// CT args: [cb_output_tiles, TensorAccessorArgs(output)...]
// RT args: [dst_addr, num_output_tiles, start_out_tile_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_out_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    const uint32_t end_out_tile_id = start_out_tile_id + num_output_tiles;
    for (uint32_t out_idx = start_out_tile_id; out_idx < end_out_tile_id; ++out_idx) {
        cb_wait_front(cb_output_tiles, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        noc_async_write_tile(out_idx, dst_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, 1);
    }
}
