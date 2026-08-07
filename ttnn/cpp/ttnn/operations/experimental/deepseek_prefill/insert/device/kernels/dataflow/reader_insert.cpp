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
//   * global_expert_idx_table[local_expert_id] < num_experts, i.e. the runtime
//     id is a legal index into the counts L1 scratch page. Out of range ⇒ this
//     core does no work at all. writer_insert.cpp makes the identical
//     decision, so the reader→writer CB stays balanced.
//   * ceil_tile(counts[id]) fits inside local_tensor — i.e. we don't over-read.
//     Enforced by clamping counts before the tile round-up, which also keeps
//     every subsequent multiplication from overflowing uint32, and again per
//     page in the read loop.
//
// Every check below is unconditional (a plain branch), with an ASSERT beside it
// so the violation is also loud in watcher builds. ASSERT alone is not enough:
// it compiles away entirely in release.
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
    // Number of experts = start/counts last dimension. Bounds the runtime
    // global_expert_id before it is used to index the L1 scratch page.
    constexpr uint32_t num_experts = get_compile_time_arg_val(7);

    constexpr uint32_t local_accessor_offset = 8;
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
    // The value comes from device memory and is not validated host-side, so it
    // must be range-checked before it indexes counts_ptr — an out-of-range id
    // reads past the scratch page and the garbage it returns becomes a DRAM
    // page id. Skip the work entirely rather than clamping to some other
    // expert's slice; writer_insert.cpp applies the identical check, so both
    // sides of the CB agree to do nothing.
    const uint32_t global_expert_id = global_expert_idx_ptr[local_expert_id];
    ASSERT(global_expert_id < num_experts);
    if (global_expert_id >= num_experts) {
        return;
    }

    // Clamp counts before the round-up below. Unclamped, a counts_value of
    // 2^32 - (TILE_HEIGHT - 1) or larger wraps in `counts_value + TILE_HEIGHT - 1`
    // and turns a huge count into a tiny one that then *passes* the assert.
    // Clamping first also bounds num_tile_rows, so neither
    // num_tile_rows * tiles_per_row nor num_tile_rows * (core_id + 1) can
    // overflow. max_counts is the largest token count the source (local_tensor)
    // can hold. Must stay byte-identical to the clamp in writer_insert.cpp so
    // the two kernels agree on how many tiles cross the CB.
    constexpr uint32_t max_tile_rows = tiles_per_row > 0 ? local_num_tiles / tiles_per_row : 0;
    constexpr uint32_t max_counts = max_tile_rows * TILE_HEIGHT;
    const uint32_t raw_counts_value = counts_ptr[global_expert_id];
    ASSERT(raw_counts_value <= max_counts);
    const uint32_t counts_value = raw_counts_value < max_counts ? raw_counts_value : max_counts;
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;

    // Runtime bounds check: slice must fit inside local_tensor. Now holds by
    // construction (see the clamp above), and is re-checked per page below.
    ASSERT(num_tiles <= local_num_tiles);

    // Split the tile rows across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // which distributes the N % num_cores remainder rows across the tail cores
    // and covers every row exactly once.
    const uint32_t my_row_start = (num_tile_rows * core_id) / num_cores;
    const uint32_t my_row_end = (num_tile_rows * (core_id + 1)) / num_cores;
    // Guard the unsigned subtraction: my_row_end < my_row_start would yield a
    // ~2^32 row count and an unbounded loop. Always false for valid inputs.
    const uint32_t my_rows = my_row_end > my_row_start ? my_row_end - my_row_start : 0;
    const uint32_t my_num_tiles = my_rows * tiles_per_row;
    const uint32_t my_start_tile = my_row_start * tiles_per_row;

    uint32_t tile_idx = my_start_tile;
    const uint32_t end_tile_idx = my_start_tile + my_num_tiles;
    while (tile_idx < end_tile_idx) {
        const uint32_t remaining = end_tile_idx - tile_idx;
        const uint32_t batch = remaining < READ_BATCH ? remaining : READ_BATCH;

        cb_tile_buf.reserve_back(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t page_id = tile_idx + i;
            // Unconditional companion to the ASSERT above: never issue a read
            // past the end of local_tensor. Always true for valid inputs. The
            // CB slot is still reserved and pushed either way, so skipping a
            // read cannot desynchronize the writer.
            if (page_id < local_num_tiles) {
                noc.async_read(
                    local_accessor,
                    cb_tile_buf,
                    cb_tile_bytes,
                    {.page_id = page_id},
                    {.offset_bytes = i * cb_tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb_tile_buf.push_back(batch);

        tile_idx += batch;
    }
}
