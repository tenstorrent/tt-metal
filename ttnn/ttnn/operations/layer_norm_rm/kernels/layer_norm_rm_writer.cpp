// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for layer_norm_rm — drains cb_output to DRAM.
//
// TILE output: pops Wt tiles per tile-row × total_tile_rows tile-rows; uses
//              raw noc_async_write_tile.
// RM output: pops 32 sticks per tile-row × total_tile_rows tile-rows; uses
//            the dataflow helper write_sticks_after_untilize semantics
//            (one stick per page, valid row_bytes per write).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t total_tile_rows = get_compile_time_arg_val(1);
    constexpr uint32_t padded_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t output_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t is_rm_output = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output = 16;
    constexpr uint32_t TILE_H = 32;

    if constexpr (is_rm_output) {
        // RM output: drain 32 sticks per tile-row, write `output_row_bytes` bytes each.
        const auto accessor = TensorAccessor(dst_args, output_addr, padded_row_bytes);
        for (uint32_t tr = 0; tr < total_tile_rows; ++tr) {
            uint32_t row_base = tr * TILE_H;
            for (uint32_t row = 0; row < TILE_H; ++row) {
                cb_wait_front(cb_output, 1);
                uint32_t l1_addr = get_read_ptr(cb_output);
                uint64_t noc_addr = accessor.get_noc_addr(row_base + row);
                noc_async_write(l1_addr, noc_addr, output_row_bytes);
                noc_async_write_barrier();
                cb_pop_front(cb_output, 1);
            }
        }
    } else {
        // TILE output: drain Wt tiles per tile-row × total_tile_rows.
        constexpr uint32_t tile_bytes_v = get_tile_size(cb_output);
        const auto accessor = TensorAccessor(dst_args, output_addr, tile_bytes_v);
        for (uint32_t tr = 0; tr < total_tile_rows; ++tr) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                uint32_t tile_id = tr * Wt + wt;
                cb_wait_front(cb_output, 1);
                uint32_t l1_addr = get_read_ptr(cb_output);
                noc_async_write_tile(tile_id, accessor, l1_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_output, 1);
            }
        }
    }
}
