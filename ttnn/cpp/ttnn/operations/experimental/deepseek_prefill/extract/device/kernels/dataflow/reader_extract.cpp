// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for the deepseek_prefill::extract op.
// Reads start[global_expert_id] and counts[global_expert_id] from DRAM at
// runtime, then streams ceil_tile(counts) * tiles_per_row tiles from
// global_tensor (starting at row/token `start`) into the reader→writer CB.
//
// Reads are issued in batches so that multiple tiles are in flight on the NOC
// at the same time (one barrier per batch instead of per tile).
//
// Bounds we check at runtime:
//   * start[id] + ceil_tile(counts[id]) fits inside global_tensor.
//   * ceil_tile(counts[id]) fits inside the output tensor (capped by
//     max_dispatched_tokens_per_expert).
//
// Bounds we DO NOT check (caller's contract):
//   * start[id] + counts[id] <= start[id + 1] — i.e. this expert's slice does
//     not overlap the next expert's slice in global_tensor. Verifying this
//     would require reading the adjacent expert's start/counts, which
//     contradicts the op's per-device contract (each device operates on a
//     single global_expert_id with no cross-expert state). Upstream code that
//     lays out the dispatch buffer is responsible for honoring this
//     invariant.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t READ_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t global_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t global_expert_idx_table_addr = get_arg_val<uint32_t>(3);
    // core_id ∈ [0, num_cores). Core i handles the tile-row range
    //   [ (N * i)     / num_cores,
    //     (N * (i+1)) / num_cores ).
    const uint32_t core_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_global_expert_idx_scratch = get_compile_time_arg_val(3);
    // Index into global_expert_idx_table. The actual global_expert_id is looked
    // up at runtime via global_expert_idx_table[local_expert_id].
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(5);
    // Upper bounds used for runtime asserts (see comment block above).
    constexpr uint32_t global_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t max_output_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);

    constexpr uint32_t global_accessor_offset = 9;
    constexpr auto global_args = TensorAccessorArgs<global_accessor_offset>();
    const auto global_accessor = TensorAccessor(global_args, global_addr, get_tile_size(cb_tile));

    constexpr uint32_t start_accessor_offset = global_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_accessor = TensorAccessor(start_args, start_addr);

    constexpr uint32_t counts_accessor_offset = start_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_accessor = TensorAccessor(counts_args, counts_addr);

    constexpr uint32_t global_expert_idx_accessor_offset = counts_args.next_compile_time_args_offset();
    constexpr auto global_expert_idx_args = TensorAccessorArgs<global_expert_idx_accessor_offset>();
    const auto global_expert_idx_accessor = TensorAccessor(global_expert_idx_args, global_expert_idx_table_addr);

    // Fetch start, counts, and global_expert_idx_table (small, 1 page each) into L1 scratch.
    const uint32_t start_l1 = get_write_ptr(cb_start_scratch);
    const uint32_t counts_l1 = get_write_ptr(cb_counts_scratch);
    const uint32_t global_expert_idx_l1 = get_write_ptr(cb_global_expert_idx_scratch);
    noc_async_read_page(0, start_accessor, start_l1);
    noc_async_read_page(0, counts_accessor, counts_l1);
    noc_async_read_page(0, global_expert_idx_accessor, global_expert_idx_l1);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_l1);
    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const volatile tt_l1_ptr uint32_t* global_expert_idx_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_expert_idx_l1);
    // Look up the runtime global_expert_id from the table at local_expert_id.
    const uint32_t global_expert_id = global_expert_idx_ptr[local_expert_id];
    const uint32_t start_value = start_ptr[global_expert_id];
    const uint32_t counts_value = counts_ptr[global_expert_id];
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;
    const uint32_t start_tile_idx = (start_value / TILE_HEIGHT) * tiles_per_row;

    // Runtime bounds checks. Note: inter-expert layout invariant
    // (start[id] + counts[id] <= start[id + 1]) is NOT enforced — see file
    // header comment.
    ASSERT(start_tile_idx + num_tiles <= global_num_tiles);
    ASSERT(num_tiles <= max_output_tiles);

    // Split the tile rows across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // which distributes the N % num_cores remainder rows across the tail cores
    // and covers every row exactly once.
    const uint32_t my_row_start = (num_tile_rows * core_id) / num_cores;
    const uint32_t my_row_end = (num_tile_rows * (core_id + 1)) / num_cores;
    const uint32_t my_rows = my_row_end - my_row_start;
    const uint32_t my_num_tiles = my_rows * tiles_per_row;
    const uint32_t my_start_tile = start_tile_idx + my_row_start * tiles_per_row;

    const uint32_t tile_bytes = get_tile_size(cb_tile);

    uint32_t tile_idx = my_start_tile;
    const uint32_t end_tile_idx = my_start_tile + my_num_tiles;
    while (tile_idx < end_tile_idx) {
        const uint32_t remaining = end_tile_idx - tile_idx;
        const uint32_t batch = remaining < READ_BATCH ? remaining : READ_BATCH;

        cb_reserve_back(cb_tile, batch);
        uint32_t l1_write_addr = get_write_ptr(cb_tile);
        for (uint32_t i = 0; i < batch; ++i) {
            noc_async_read_tile(tile_idx + i, global_accessor, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_tile, batch);

        tile_idx += batch;
    }
}
