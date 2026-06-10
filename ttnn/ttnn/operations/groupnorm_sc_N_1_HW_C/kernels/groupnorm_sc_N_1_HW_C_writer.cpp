// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for groupnorm_sc_N_1_HW_C: drains cb_output_tiles group by group and
// writes each tile back to DRAM at index n*Ht*Wt + r*Wt + g*Wg + c (same
// formula the reader uses — output shape == input shape). This core handles
// group ids [start_group, start_group + num_groups_here), n = id / G, g = id % G.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_group = get_arg_val<uint32_t>(1);
    const uint32_t num_groups_here = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Wg = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_output_tiles = 16;

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);

    // Batched per row chunk (Wg tiles, one barrier) — cb_output_tiles is sized
    // 2*Wg pages so chunk batching preserves the double-buffered overlap.
    uint32_t n = start_group / G;
    uint32_t g = start_group % G;
    for (uint32_t i = 0; i < num_groups_here; ++i) {
        const uint32_t group_base = n * Ht * Wt + g * Wg;
        for (uint32_t r = 0; r < Ht; ++r) {
            const uint32_t row_base = group_base + r * Wt;
            cb_wait_front(cb_output_tiles, Wg);
            uint32_t l1_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t c = 0; c < Wg; ++c) {
                noc_async_write_tile(row_base + c, output, l1_addr);
                l1_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, Wg);
        }
        if (++g == G) {
            g = 0;
            ++n;
        }
    }
}
