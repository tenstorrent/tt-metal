// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// atan_mean — Writer kernel.
//
// Drains one float32 output tile per row-tile from ``cb_output_tiles`` and
// writes it to DRAM. The output buffer is allocated as ``(N, C, H, 1)``
// TILE_LAYOUT — physically padded to ``(N, C, H, 32)``, one W-tile per
// row-tile — so the output tile id equals the global row-tile index:
//     output_tile_id = r
//
// CT args: [CB_OUTPUT_TILES, TensorAccessorArgs...]
// RT args: [dst_addr, num_row_tiles, start_row_tile]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_row_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    const uint32_t end_row_tile = start_row_tile + num_row_tiles;
    for (uint32_t r = start_row_tile; r < end_row_tile; ++r) {
        cb_wait_front(cb_output_tiles, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        noc_async_write_tile(r, dst_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, 1);
    }
}
