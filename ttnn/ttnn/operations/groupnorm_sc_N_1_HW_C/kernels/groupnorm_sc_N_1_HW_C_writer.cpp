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
    // Cluster path (Refinement 3): work unit = (n, cluster), one Ht x Wcu write.
    constexpr bool GROUPS_NA = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t WC_FULL = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_CLUSTERS = get_compile_time_arg_val(6);
    constexpr uint32_t C = get_compile_time_arg_val(7);
    constexpr uint32_t CLUSTER_CH = get_compile_time_arg_val(8);
    constexpr auto output_args = TensorAccessorArgs<9>();

    constexpr uint32_t cb_output_tiles = 16;

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);

    // Batched per row chunk (one barrier) — cb_output_tiles is sized 2x the
    // widest chunk so chunk batching preserves the double-buffered overlap.
    auto write_rows = [&](uint32_t n, uint32_t col0, uint32_t width) {
        const uint32_t base = n * Ht * Wt + col0;
        for (uint32_t r = 0; r < Ht; ++r) {
            const uint32_t row_base = base + r * Wt;
            cb_wait_front(cb_output_tiles, width);
            uint32_t l1_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t c = 0; c < width; ++c) {
                noc_async_write_tile(row_base + c, output, l1_addr);
                l1_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, width);
        }
    };

    if constexpr (!GROUPS_NA) {
        uint32_t n = start_group / G;
        uint32_t g = start_group % G;
        for (uint32_t i = 0; i < num_groups_here; ++i) {
            write_rows(n, g * Wg, Wg);
            if (++g == G) {
                g = 0;
                ++n;
            }
        }
    } else {
        uint32_t n = start_group / NUM_CLUSTERS;
        uint32_t cl = start_group % NUM_CLUSTERS;
        for (uint32_t i = 0; i < num_groups_here; ++i) {
            const uint32_t cluster_c0 = cl * CLUSTER_CH;
            const uint32_t Ccl = (C - cluster_c0 < CLUSTER_CH) ? (C - cluster_c0) : CLUSTER_CH;
            const uint32_t Wcu = (Ccl + 31) / 32;
            write_rows(n, cl * WC_FULL, Wcu);
            if (++cl == NUM_CLUSTERS) {
                cl = 0;
                ++n;
            }
        }
    }
}
