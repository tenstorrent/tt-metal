// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — candidate A compute: splits the block's OUTPUT
// tile-COLUMNS (not rows) across two output CBs, one per data-movement RISC-V.
//
// WHY COLUMNS AND NOT ROWS. The op's existing SPLIT READER mirrors this idea
// on the input side by splitting ROWS, because a reader's atomic unit is a
// tile-ROW (one stick-run push). But on the focus shape `block_row_extent`
// (R) is exactly 1 — a single tile-row of `block_width_tiles` (8) columns —
// so a row split has NOTHING to split there: 1 row cannot become two
// non-empty row groups. The write cost lives entirely INSIDE that one row's
// 8-tile store loop, so the split has to cut the row itself, across tile
// columns, to have any effect on the shape this idea was raised against.
//
// MECHANISM. `cb_input_rows` still has exactly ONE producer (the reader) and
// ONE consumer (this kernel) — the reader pushes the LEFT half-width columns
// of every tile-row first (all `block_row_extent` rows), then the RIGHT
// half-width columns of every tile-row (see reader_col_split.cpp), so the
// two `tilize<>` calls below consume the CB in the exact order it was filled:
// first call pops every row's LEFT half, second call pops every row's RIGHT
// half. Each call targets a DIFFERENT output CB
// (`cb_output_tiles` / `cb_output_tiles_split`), so each output CB keeps its
// own single-producer/single-consumer contract — the hard CB invariant this
// idea has to design around, not paper over.
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib::tilize_config;

    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);        // LEFT half (BRISC's)
    constexpr uint32_t cb_output_tiles_split = get_compile_time_arg_val(2);  // RIGHT half (NCRISC's)
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(5);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t half_width_tiles = block_width_tiles / 2;

    const uint32_t start_block_id = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);
    const uint32_t block_stride = get_arg_val<uint32_t>(2);

    {
        MaybeDeviceZoneScope("compute_startup");
        // Programs pack format from cb_output_tiles. cb_output_tiles_split
        // carries the IDENTICAL data format (same output dtype), and every
        // tilize<> call below reconfigures explicitly
        // (UnpackAndPackReconfigure), so the second call's target-CB switch
        // is never silently skipped.
        compute_kernel_hw_startup(cb_input_rows, cb_output_tiles);
    }

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        {
            MaybeDeviceZoneScope("compute_col_split_left");
            compute_kernel_lib::tilize<
                half_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                InitUninitMode::InitAndUninit,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                Fp32Mode::Fast>(block_row_extent);
        }
        {
            MaybeDeviceZoneScope("compute_col_split_right");
            compute_kernel_lib::tilize<
                half_width_tiles,
                cb_input_rows,
                cb_output_tiles_split,
                InitUninitMode::InitAndUninit,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                Fp32Mode::Fast>(block_row_extent);
        }
    }
}
