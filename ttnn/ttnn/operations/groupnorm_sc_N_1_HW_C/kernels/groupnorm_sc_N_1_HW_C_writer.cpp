// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for groupnorm_sc_N_1_HW_C: drains cb_output_tiles group by group and
// writes each tile back to DRAM at index n*Ht*Wt + r*Wt + g*Wg + c (same
// formula the reader uses — output shape == input shape).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Wg = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    constexpr uint32_t cb_output_tiles = 16;

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t group_base = n * Ht * Wt + g * Wg;
            for (uint32_t r = 0; r < Ht; ++r) {
                for (uint32_t c = 0; c < Wg; ++c) {
                    const uint32_t tile_id = group_base + r * Wt + c;
                    cb_wait_front(cb_output_tiles, 1);
                    noc_async_write_tile(tile_id, output, get_read_ptr(cb_output_tiles));
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_tiles, 1);
                }
            }
        }
    }
}
