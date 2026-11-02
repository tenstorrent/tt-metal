// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Drains every 2nd output row (rows first_row, first_row + 2, ...) of the epilogue bench; stores them to
// the output tensor on the last iteration only (stores stubbed elsewhere, as in the coordinator's ablation).
// expose_inputs: publish the resident (sharded-backed) partials / table CBs once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t iters = get_compile_time_arg_val(3);
    constexpr uint32_t first_row = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t expose_inputs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_partials = get_compile_time_arg_val(7);
    constexpr uint32_t partials_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t cb_table = get_compile_time_arg_val(9);
    constexpr uint32_t table_tiles = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    if constexpr (expose_inputs) {
        cb_reserve_back(cb_partials, partials_tiles);
        cb_push_back(cb_partials, partials_tiles);
        cb_reserve_back(cb_table, table_tiles);
        cb_push_back(cb_table, table_tiles);
    }
    for (uint32_t iter = 0; iter < iters; ++iter) {
        for (uint32_t row = first_row; row < block_m; row += 2) {
            cb_wait_front(cb_out, n_tiles);
            if (iter + 1 == iters) {
                uint32_t l1_addr = get_read_ptr(cb_out);
                for (uint32_t n = 0; n < n_tiles; ++n) {
                    noc_async_write_page(row * n_tiles + n, out_accessor, l1_addr);
                    l1_addr += tile_bytes;
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_out, n_tiles);
        }
    }
    noc_async_write_barrier();
}
