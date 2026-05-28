// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — Writer kernel (BRISC)
//
// Drains cb_output_tiles in (n, T, r) order to DRAM, matching the compute
// kernel's push order. Output tensor is TILE_LAYOUT with N * Ct * Ht tiles.
// Each tile is written via TensorAccessor + noc_async_write_tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    constexpr uint32_t CB_OUTPUT_TILES = 16;
    uint32_t tile_bytes = get_tile_size(CB_OUTPUT_TILES);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Output tile push order from compute (must match compute kernel exactly):
    //   for n in 0..N-1:
    //     for T in 0..Ct-1:
    //       for r in 0..Ht-1:
    //         push 1 output tile
    // page_id = n * (Ht * Ct) + r * Ct + T  (matches TILE_LAYOUT (N, 1, Ht, Ct))
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t T = 0; T < Ct; ++T) {
            for (uint32_t r = 0; r < Ht; ++r) {
                cb_wait_front(CB_OUTPUT_TILES, 1);
                uint32_t l1_addr = get_read_ptr(CB_OUTPUT_TILES);
                uint32_t page_id = n * (Ht * Ct) + r * Ct + T;
                noc_async_write_tile(page_id, accessor, l1_addr);
                noc_async_write_barrier();
                cb_pop_front(CB_OUTPUT_TILES, 1);
            }
        }
    }
}
