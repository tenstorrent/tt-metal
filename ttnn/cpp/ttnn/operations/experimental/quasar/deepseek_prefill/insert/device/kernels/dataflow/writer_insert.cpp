// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for the deepseek_prefill::insert op.
// Reads start[global_expert_id] and counts[global_expert_id] from DRAM at
// runtime, then drains ceil_tile(counts) * tiles_per_row tiles from the
// reader→writer DFB into global_tensor starting at tile index
// (start / TILE_HEIGHT) * tiles_per_row.
//
// Writes are issued in batches so multiple tiles are in flight on the NOC at
// the same time (one barrier per batch instead of per tile).
//
// Bounds we check at runtime:
//   * start[id] + ceil_tile(counts[id]) fits inside global_tensor.
//
// Bounds we DO NOT check (caller's contract):
//   * start[id] + counts[id] <= start[id + 1] — see file header in
//     insert_device_operation.cpp for rationale.

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
    // core_id ∈ [0, num_cores). Core i writes the global tile-row range
    //   [ start_row + (N * i)     / num_cores,
    //     start_row + (N * (i+1)) / num_cores ).
    const uint32_t core_id = get_arg(args::core_id);

    Noc noc;
    DataflowBuffer dfb_tile(dfb::tile);
    Scratchpad<uint32_t> start_scratch(scratch::start);
    Scratchpad<uint32_t> counts_scratch(scratch::counts);
    Scratchpad<uint32_t> global_expert_idx_scratch(scratch::global_expert_idx);
    const uint32_t dfb_tile_bytes = dfb_tile.get_tile_size();
    // Index into global_expert_idx_table. The actual global_expert_id is looked
    // up at runtime via global_expert_idx_table[local_expert_id].
    constexpr uint32_t local_expert_id = get_arg(args::local_expert_id);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t global_num_tiles = get_arg(args::global_num_tiles);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const auto global_accessor = TensorAccessor(tensor::global);
    const auto start_accessor = TensorAccessor(tensor::start);
    const auto counts_accessor = TensorAccessor(tensor::counts);
    const auto global_expert_idx_accessor = TensorAccessor(tensor::global_expert_idx_table);

    // Fetch start, counts, and global_expert_idx_table (small, 1 page each) into L1 scratch.
    noc.async_read(
        start_accessor, start_scratch, start_accessor.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(
        counts_accessor, counts_scratch, counts_accessor.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(
        global_expert_idx_accessor,
        global_expert_idx_scratch,
        global_expert_idx_accessor.get_aligned_page_size(),
        {.page_id = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    const volatile tt_l1_ptr uint32_t* start_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(start_scratch));
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(counts_scratch));
    const volatile tt_l1_ptr uint32_t* global_expert_idx_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(global_expert_idx_scratch));
    // Look up the runtime global_expert_id from the table at local_expert_id.
    const uint32_t global_expert_id = global_expert_idx_ptr[local_expert_id];
    const uint32_t start_value = start_ptr[global_expert_id];
    const uint32_t counts_value = counts_ptr[global_expert_id];
    const uint32_t counts_rounded_up = ((counts_value + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    const uint32_t num_tile_rows = counts_rounded_up / TILE_HEIGHT;
    const uint32_t num_tiles = num_tile_rows * tiles_per_row;
    const uint32_t start_tile_idx = (start_value / TILE_HEIGHT) * tiles_per_row;

    // Runtime bounds check: slice must stay inside global_tensor.
    ASSERT(start_tile_idx + num_tiles <= global_num_tiles);

    // Split the tile rows across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // matching the reader so this core's DFB is drained by exactly its own
    // writer. The destination tile index in global is offset by
    // start_tile_idx + my_row_start * tiles_per_row.
    const uint32_t my_row_start = (num_tile_rows * core_id) / num_cores;
    const uint32_t my_row_end = (num_tile_rows * (core_id + 1)) / num_cores;
    const uint32_t my_rows = my_row_end - my_row_start;
    const uint32_t my_num_tiles = my_rows * tiles_per_row;
    const uint32_t my_dst_start = start_tile_idx + my_row_start * tiles_per_row;

    uint32_t offset = 0;
    while (offset < my_num_tiles) {
        const uint32_t remaining = my_num_tiles - offset;
        const uint32_t batch = remaining < WRITE_BATCH ? remaining : WRITE_BATCH;

        dfb_tile.wait_front(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            noc.async_write(
                dfb_tile,
                global_accessor,
                dfb_tile_bytes,
                {.offset_bytes = i * dfb_tile_bytes},
                {.page_id = my_dst_start + offset + i});
        }
        noc.async_write_barrier();
        dfb_tile.pop_front(batch);

        offset += batch;
    }
}
