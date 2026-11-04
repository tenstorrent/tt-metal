// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_dispatch compute: tilize_block per block. Each block's 32 gathered rows x w tiles sit in one slot of the
// aliased rows CB: rows [0, rows_noc0) pushed on cb_rows_head by reader_noc0, the rest on cb_rows_tail by reader_noc1.
// The caller waits on the tail credit; compute_kernel_lib::tilize waits / pops the head credit and reserves / pushes
// w tiles on cb_output_tiles, which is backed on this core's output shard (slot j = block j). Tilize init happens once
// (with the first block) and uninit once (with the last block): the helper asserts num_blocks > 0, so the design's
// zero-block InitOnly / UninitOnly calls are folded into the first / last block's call.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

template <
    uint32_t w,
    uint32_t cb_rows_head,
    uint32_t cb_output_tiles,
    InitUninitMode init_mode,
    ReconfigureRegisterDatatypeMode reconfig,
    typename Buffer>
ALWI void tilize_block(Buffer& rows_head, Buffer& rows_tail, Buffer& output_tiles) {
    rows_tail.wait_front(w);  // reader_noc1's half of the slot
    compute_kernel_lib::tilize<w, cb_rows_head, cb_output_tiles, init_mode, WaitMode::WaitBlock, reconfig>(
        rows_head, output_tiles, 1);
    rows_tail.pop_front(w);
}

void kernel_main() {
    constexpr uint32_t w = get_named_compile_time_arg_val("block_width_tiles");
    constexpr uint32_t num_shards = get_named_compile_time_arg_val("num_shards");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t cb_rows_head = get_named_compile_time_arg_val("cb_rows_head");
    constexpr uint32_t cb_rows_tail = get_named_compile_time_arg_val("cb_rows_tail");
    constexpr uint32_t cb_output_tiles = get_named_compile_time_arg_val("cb_output_tiles");

    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    if (core >= num_shards) {
        return;  // no shard on this core
    }
    const uint32_t num_blocks = (num_shards - core + num_cores - 1) / num_cores;

    compute_kernel_hw_startup(cb_rows_head, cb_output_tiles);
    DataflowBuffer rows_head(cb_rows_head);
    DataflowBuffer rows_tail(cb_rows_tail);
    DataflowBuffer output_tiles(cb_output_tiles);

    constexpr auto reconfig = ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure;
    constexpr auto keep = ReconfigureRegisterDatatypeMode::NoReconfigure;  // same formats as the first block
    if (num_blocks == 1) {
        tilize_block<w, cb_rows_head, cb_output_tiles, InitUninitMode::InitAndUninit, reconfig>(
            rows_head, rows_tail, output_tiles);
        return;
    }
    tilize_block<w, cb_rows_head, cb_output_tiles, InitUninitMode::InitOnly, reconfig>(
        rows_head, rows_tail, output_tiles);
    for (uint32_t block_idx = 1; block_idx + 1 < num_blocks; ++block_idx) {
        tilize_block<w, cb_rows_head, cb_output_tiles, InitUninitMode::Neither, keep>(
            rows_head, rows_tail, output_tiles);
    }
    tilize_block<w, cb_rows_head, cb_output_tiles, InitUninitMode::UninitOnly, keep>(
        rows_head, rows_tail, output_tiles);
}
