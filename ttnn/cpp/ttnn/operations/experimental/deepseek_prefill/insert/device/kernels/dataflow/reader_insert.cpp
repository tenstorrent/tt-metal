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
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t READ_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t global_expert_id = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t local_num_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t local_accessor_offset = 5;
    constexpr auto local_args = TensorAccessorArgs<local_accessor_offset>();
    const auto local_accessor = TensorAccessor(local_args, local_addr, get_tile_size(cb_tile));

    constexpr uint32_t counts_accessor_offset = local_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_accessor = TensorAccessor(counts_args, counts_addr);

    // Fetch the counts tensor (small, 1 page) into L1 scratch.
    const uint32_t counts_l1 = get_write_ptr(cb_counts_scratch);
    noc_async_read_page(0, counts_accessor, counts_l1);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const uint32_t counts_value = counts_ptr[global_expert_id];
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;

    // Runtime bounds check: slice must fit inside local_tensor.
    ASSERT(num_tiles <= local_num_tiles);

    const uint32_t tile_bytes = get_tile_size(cb_tile);

    uint32_t tile_idx = 0;
    while (tile_idx < num_tiles) {
        const uint32_t remaining = num_tiles - tile_idx;
        const uint32_t batch = remaining < READ_BATCH ? remaining : READ_BATCH;

        cb_reserve_back(cb_tile, batch);
        uint32_t l1_write_addr = get_write_ptr(cb_tile);
        for (uint32_t i = 0; i < batch; ++i) {
            noc_async_read_tile(tile_idx + i, local_accessor, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_tile, batch);

        tile_idx += batch;
    }
}
