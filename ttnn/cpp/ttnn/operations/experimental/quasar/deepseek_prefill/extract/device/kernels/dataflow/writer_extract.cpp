// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for the deepseek_prefill::extract op.
// Reads counts[global_expert_id] from DRAM at runtime to determine how many
// tiles to drain from the reader→writer DFB and write to the output tensor.
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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "api/debug/assert.h"
#include "experimental/kernel_args.h"

constexpr uint32_t TILE_HEIGHT = 32;  // allow-redefined-helper: verbatim Quasar copy of the original op
constexpr uint32_t WRITE_BATCH = 8;   // tiles per NOC barrier; must be <= DFB depth.

// L1 address the CPU accesses a scratchpad through: on Quasar DM the uncached alias, as DataflowBuffer pointers are,
// since the CPU caches are not coherent with NoC writes to L1; on WH/BH the plain address.
template <typename T>
uint32_t cpu_l1_addr(const Scratchpad<T>& scratch) {
    uintptr_t addr = local_mem::detail::AddressAccess::address(scratch.local_mem());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    return static_cast<uint32_t>(addr);
}

void kernel_main() {
    // core_id ∈ [0, num_cores). Core i writes to output tile-row range
    //   [ (N * i)     / num_cores,
    //     (N * (i+1)) / num_cores ).
    const uint32_t core_id = get_arg(args::core_id);

    Noc noc;
    DataflowBuffer dfb_tile(dfb::tile);
    Scratchpad<uint32_t> counts_scratch(scratch::counts);
    Scratchpad<uint32_t> global_expert_idx_scratch(scratch::global_expert_idx);
    const uint32_t dfb_tile_bytes = dfb_tile.get_tile_size();
    // Index into global_expert_idx_table. The actual global_expert_id is looked
    // up at runtime via global_expert_idx_table[local_expert_id].
    constexpr uint32_t local_expert_id = get_arg(args::local_expert_id);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    // Upper bound used for runtime assert (see comment block above).
    constexpr uint32_t max_output_tiles = get_arg(args::max_output_tiles);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const auto output_accessor = TensorAccessor(tensor::output);
    const auto counts_accessor = TensorAccessor(tensor::counts);
    const auto global_expert_idx_accessor = TensorAccessor(tensor::global_expert_idx_table);

    // Fetch counts and global_expert_idx_table (small, 1 page each) into L1 scratch.
    noc.async_read(
        counts_accessor, counts_scratch, counts_accessor.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(
        global_expert_idx_accessor,
        global_expert_idx_scratch,
        global_expert_idx_accessor.get_aligned_page_size(),
        {.page_id = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(counts_scratch));
    const volatile tt_l1_ptr uint32_t* global_expert_idx_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(global_expert_idx_scratch));
    // Look up the runtime global_expert_id from the table at local_expert_id.
    const uint32_t global_expert_id = global_expert_idx_ptr[local_expert_id];
    const uint32_t counts_value = counts_ptr[global_expert_id];
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;

    // Runtime bounds check. Note: inter-expert layout invariant
    // (start[id] + counts[id] <= start[id + 1]) is NOT enforced — see file
    // header comment.
    ASSERT(num_tiles <= max_output_tiles);

    // Split the tile rows across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // matching the reader so this core's DFB is drained by exactly its own
    // writer.
    const uint32_t my_row_start = (num_tile_rows * core_id) / num_cores;
    const uint32_t my_row_end = (num_tile_rows * (core_id + 1)) / num_cores;
    const uint32_t my_rows = my_row_end - my_row_start;
    const uint32_t my_num_tiles = my_rows * tiles_per_row;
    const uint32_t my_start_tile = my_row_start * tiles_per_row;

    uint32_t tile_idx = my_start_tile;
    const uint32_t end_tile_idx = my_start_tile + my_num_tiles;
    while (tile_idx < end_tile_idx) {
        const uint32_t remaining = end_tile_idx - tile_idx;
        const uint32_t batch = remaining < WRITE_BATCH ? remaining : WRITE_BATCH;

        dfb_tile.wait_front(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            noc.async_write(
                dfb_tile,
                output_accessor,
                dfb_tile_bytes,
                {.offset_bytes = i * dfb_tile_bytes},
                {.page_id = tile_idx + i});
        }
        noc.async_write_barrier();
        dfb_tile.pop_front(batch);

        tile_idx += batch;
    }
}
