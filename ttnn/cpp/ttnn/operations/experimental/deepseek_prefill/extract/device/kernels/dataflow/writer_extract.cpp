// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for the deepseek_prefill::extract op.
// Reads counts[global_expert_id] from DRAM at runtime to determine how many
// tiles to drain from the reader→writer CB and write to the output tensor.
//
// Writes are issued in batches so multiple tiles are in flight on the NOC at
// the same time (one barrier per batch instead of per tile).
//
// Bounds we check at runtime:
//   * ceil_tile(counts[id]) fits inside the output tensor (capped by
//     max_dispatched_tokens_per_expert).
//
// Bounds we DO NOT check (caller's contract):
//   * start[id] + counts[id] <= start[id + 1] — i.e. this expert's slice does
//     not overlap the next expert's slice in global_tensor. The writer doesn't
//     touch global_tensor so this bound is the reader's domain; see
//     reader_extract.cpp for the full rationale.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t WRITE_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t global_expert_id = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);
    // Upper bound used for runtime assert (see comment block above).
    constexpr uint32_t max_output_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t output_accessor_offset = 5;
    constexpr auto output_args = TensorAccessorArgs<output_accessor_offset>();
    const auto output_accessor = TensorAccessor(output_args, output_addr, get_tile_size(cb_tile));

    constexpr uint32_t counts_accessor_offset = output_args.next_compile_time_args_offset();
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

    // Runtime bounds check. Note: inter-expert layout invariant
    // (start[id] + counts[id] <= start[id + 1]) is NOT enforced — see file
    // header comment.
    ASSERT(num_tiles <= max_output_tiles);

    const uint32_t tile_bytes = get_tile_size(cb_tile);

    uint32_t tile_idx = 0;
    while (tile_idx < num_tiles) {
        const uint32_t remaining = num_tiles - tile_idx;
        const uint32_t batch = remaining < WRITE_BATCH ? remaining : WRITE_BATCH;

        cb_wait_front(cb_tile, batch);
        uint32_t l1_read_addr = get_read_ptr(cb_tile);
        for (uint32_t i = 0; i < batch; ++i) {
            noc_async_write_tile(tile_idx + i, output_accessor, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_tile, batch);

        tile_idx += batch;
    }
}
