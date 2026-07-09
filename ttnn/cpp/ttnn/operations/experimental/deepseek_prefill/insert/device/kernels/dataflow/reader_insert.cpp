// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for the deepseek_prefill::insert op.
// Reads counts[global_expert_id] from DRAM at runtime, then streams
// ceil_tile(counts) * tiles_per_row tiles from the *front* of local_tensor
// into the reader→writer CB.
//
// Reads are issued in batches so that multiple tiles are in flight on the NOC
// at the same time (one barrier per batch instead of per tile).
//
// Bounds we check at runtime:
//   * ceil_tile(counts[id]) fits inside local_tensor — i.e. we don't over-read.
//
// Bounds we DO NOT check (caller's contract):
//   * start[id] + counts[id] <= start[id + 1] — see file header in
//     insert_device_operation.cpp for rationale.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t READ_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t global_expert_idx_table_addr = get_arg_val<uint32_t>(2);
    // core_id ∈ [0, num_cores). Core i reads the local tile-row range
    //   [ (N * i)     / num_cores,
    //     (N * (i+1)) / num_cores ).
    const uint32_t core_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_global_expert_idx_scratch = get_compile_time_arg_val(2);

    Noc noc;
    CircularBuffer cb_tile_buf(cb_tile);
    CircularBuffer cb_counts_scratch_buf(cb_counts_scratch);
    CircularBuffer cb_global_expert_idx_scratch_buf(cb_global_expert_idx_scratch);
    const uint32_t cb_tile_bytes = cb_tile_buf.get_tile_size();
    // Index into global_expert_idx_table. The actual global_expert_id is looked
    // up at runtime via global_expert_idx_table[local_expert_id].
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t local_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);

    constexpr uint32_t local_accessor_offset = 7;
    constexpr auto local_args = TensorAccessorArgs<local_accessor_offset>();
    const auto local_accessor = TensorAccessor(local_args, local_addr, get_tile_size(cb_tile));

    constexpr uint32_t counts_accessor_offset = local_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_accessor = TensorAccessor(counts_args, counts_addr);

    constexpr uint32_t global_expert_idx_accessor_offset = counts_args.next_compile_time_args_offset();
    constexpr auto global_expert_idx_args = TensorAccessorArgs<global_expert_idx_accessor_offset>();
    const auto global_expert_idx_accessor = TensorAccessor(global_expert_idx_args, global_expert_idx_table_addr);

    // Fetch counts and global_expert_idx_table (small, 1 page each) into L1 scratch.
    noc.async_read(
        counts_accessor,
        cb_counts_scratch_buf,
        counts_accessor.get_aligned_page_size(),
        {.page_id = 0},
        {.offset_bytes = 0});
    noc.async_read(
        global_expert_idx_accessor,
        cb_global_expert_idx_scratch_buf,
        global_expert_idx_accessor.get_aligned_page_size(),
        {.page_id = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_counts_scratch_buf.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* global_expert_idx_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_global_expert_idx_scratch_buf.get_write_ptr());
    // Look up the runtime global_expert_id from the table at local_expert_id.
    const uint32_t global_expert_id = global_expert_idx_ptr[local_expert_id];
    const uint32_t counts_value = counts_ptr[global_expert_id];
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;

    // Runtime bounds check: slice must fit inside local_tensor.
    ASSERT(num_tiles <= local_num_tiles);

    // Split the tile rows across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // which distributes the N % num_cores remainder rows across the tail cores
    // and covers every row exactly once.
    const uint32_t my_row_start = (num_tile_rows * core_id) / num_cores;
    const uint32_t my_row_end = (num_tile_rows * (core_id + 1)) / num_cores;
    const uint32_t my_rows = my_row_end - my_row_start;
    const uint32_t my_num_tiles = my_rows * tiles_per_row;
    const uint32_t my_start_tile = my_row_start * tiles_per_row;

    uint32_t tile_idx = my_start_tile;
    const uint32_t end_tile_idx = my_start_tile + my_num_tiles;
    while (tile_idx < end_tile_idx) {
        const uint32_t remaining = end_tile_idx - tile_idx;
        const uint32_t batch = remaining < READ_BATCH ? remaining : READ_BATCH;

        cb_tile_buf.reserve_back(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            noc.async_read(
                local_accessor,
                cb_tile_buf,
                cb_tile_bytes,
                {.page_id = tile_idx + i},
                {.offset_bytes = i * cb_tile_bytes});
        }
        noc.async_read_barrier();
        cb_tile_buf.push_back(batch);

        tile_idx += batch;
    }
}
