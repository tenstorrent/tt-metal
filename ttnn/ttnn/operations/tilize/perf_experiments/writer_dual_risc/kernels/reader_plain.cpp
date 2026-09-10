// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — PLAIN reader (NCRISC / default NoC0).
//
// Isolated reconstruction of the op's current "load_block" (the plain,
// non-split, non-padded, non-retile, non-native, non-strided branch). Held
// IDENTICAL across every variant in this bench (baseline, col_split,
// role_swap) so the measured delta is attributable ONLY to the writer side.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(6);
    constexpr auto in_args = TensorAccessorArgs<7>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        MaybeDeviceZoneScope("reader_read_block");
        dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, dataflow_kernel_lib::TilizeGranularity::TILE>(
            in_acc,
            /* total_num_rows          */ block_row_extent * tile_h,
            /* row_bytes               */ block_row_bytes,
            /* start_page              */ row_start * tile_h,
            /* byte_offset_within_page */ w_chunk * block_row_bytes);
    }
}
