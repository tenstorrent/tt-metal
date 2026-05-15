// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for rms_norm.
//
// Two paths:
//   - OUTPUT_IS_RM:  consume Wt tile-sized pages per chunk from cb_output_rm
//                    via write_sticks_after_untilize (helper handles partial-H
//                    by writing only `total_units` valid rows).
//   - !OUTPUT_IS_RM: consume Wt tile-sized pages per chunk from cb_output_tiles,
//                    issuing per-tile noc_async_write_tile.
//
// Writer runs on BRISC (RISCV_0). It pops from CBs and never pushes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t OUTPUT_IS_RM = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t output_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    // ---- Runtime args ----
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t total_units = get_arg_val<uint32_t>(2);

    // ---- CB ids ----
    constexpr uint32_t cb_output_tiles = 16;
    constexpr uint32_t cb_output_rm = 17;

    if constexpr (OUTPUT_IS_RM) {
        // total_units = total valid rows (last chunk may be partial). The helper
        // pops Wt tile-sized pages per 32-row block and writes only valid rows.
        const auto accessor = TensorAccessor(dst_args, dst_addr);
        dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(
            accessor, total_units, output_row_bytes, start_unit);
    } else {
        // TILE output: stream tiles directly. total_units = NC*Ht*Wt.
        const uint32_t output_tile_bytes = get_tile_size(cb_output_tiles);
        const auto accessor = TensorAccessor(dst_args, dst_addr, output_tile_bytes);
        for (uint32_t i = 0; i < total_units; ++i) {
            cb_wait_front(cb_output_tiles, 1);
            const uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            noc_async_write_tile(start_unit + i, accessor, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, 1);
        }
    }
}
