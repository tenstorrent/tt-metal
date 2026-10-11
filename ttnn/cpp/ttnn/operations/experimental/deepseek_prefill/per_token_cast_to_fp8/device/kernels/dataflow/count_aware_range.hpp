// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Token-count-aware work range for per_token_cast_to_fp8 over a MoE expert buffer (TILE input).
// The valid prefix of the flat expert-major buffer is max over this chip's local experts of
// region_offset[g] + ceil32(count[g]) (g = global_expert_idx_table[local]). Both dataflow kernels
// compute it on device from the same three small tables, then take a balanced slice of the
// (row-tile x 128-chunk) block space by core_id. Mirrors per_token_cast_back's token_count_aware path.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

struct CountAwareRange {
    uint32_t first_block;
    uint32_t num_blocks;
};

// Valid tile-rows of an expert buffer: max over this chip's experts of region_offset + ceil(count, tile_h),
// clamped to max_rows, in tile-rows. Reads the three routing tables once.
template <typename RegionT, typename CountsT, typename TableT>
inline uint32_t count_aware_valid_tile_rows(
    Noc& noc,
    const RegionT& region,
    const CountsT& counts,
    const TableT& table,
    CircularBuffer& region_scratch,
    CircularBuffer& counts_scratch,
    CircularBuffer& table_scratch,
    uint32_t experts_per_chip,
    uint32_t max_rows,
    uint32_t tile_h) {
    region_scratch.reserve_back(1);
    counts_scratch.reserve_back(1);
    table_scratch.reserve_back(1);
    noc.async_read(region, region_scratch, region.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(counts, counts_scratch, counts.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(table, table_scratch, table.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const volatile tt_l1_ptr uint32_t* region_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region_scratch.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_scratch.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* table_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_scratch.get_write_ptr());
    uint32_t valid_rows = 0;
    for (uint32_t local = 0; local < experts_per_chip; ++local) {
        const uint32_t g = table_ptr[local];
        const uint32_t end = region_ptr[g] + ((counts_ptr[g] + tile_h - 1) / tile_h) * tile_h;
        if (end > valid_rows) {
            valid_rows = end;
        }
    }
    if (valid_rows > max_rows) {
        valid_rows = max_rows;
    }
    return (valid_rows + tile_h - 1) / tile_h;
}

// Valid tile-rows, read once per chip: the root core reads the three tables and multicasts valid_tile_rows + 1 into
// every core's semaphore (loopback included); the others poll theirs. 240 kernels reading the same three DRAM pages
// queued on one bank and delayed the first unit by up to ~15 us at K3 load.
template <typename RegionT, typename CountsT, typename TableT>
inline uint32_t count_aware_valid_tile_rows_mcast(
    Noc& noc,
    const RegionT& region,
    const CountsT& counts,
    const TableT& table,
    CircularBuffer& region_scratch,
    CircularBuffer& counts_scratch,
    CircularBuffer& table_scratch,
    uint32_t experts_per_chip,
    uint32_t max_rows,
    uint32_t tile_h,
    bool is_root,
    uint32_t sem_addr,
    uint64_t mcast_sem_noc_addr,
    uint32_t num_dests) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    if (is_root) {
        const uint32_t rows = count_aware_valid_tile_rows(
            noc,
            region,
            counts,
            table,
            region_scratch,
            counts_scratch,
            table_scratch,
            experts_per_chip,
            max_rows,
            tile_h);
        *sem = rows + 1;
        noc_semaphore_set_multicast_loopback_src(sem_addr, mcast_sem_noc_addr, num_dests);
        noc_async_write_barrier();
        return rows;
    }
    noc_semaphore_wait_min(sem, 1);
    return *sem - 1;
}

// This core's share [frac_start, frac_end) (1/65536 units) of valid_tile_rows * chunks_per_row blocks.
inline CountAwareRange count_aware_block_range_frac_rows(
    uint32_t valid_tile_rows, uint32_t chunks_per_row, uint32_t frac_start, uint32_t frac_end) {
    const uint64_t total_blocks = static_cast<uint64_t>(valid_tile_rows) * chunks_per_row;
    const uint32_t start = static_cast<uint32_t>((total_blocks * frac_start) >> 16);
    const uint32_t end = static_cast<uint32_t>((total_blocks * frac_end) >> 16);
    return {start, end - start};
}

// This core's even share (core_id of num_cores) of valid_tile_rows * chunks_per_row blocks.
inline CountAwareRange count_aware_block_range_rows(
    uint32_t valid_tile_rows, uint32_t chunks_per_row, uint32_t core_id, uint32_t num_cores) {
    const uint32_t total_blocks = valid_tile_rows * chunks_per_row;
    const uint32_t start = (total_blocks * core_id) / num_cores;
    const uint32_t end = (total_blocks * (core_id + 1)) / num_cores;
    return {start, end - start};
}

// Same, but this core's share is [frac_start, frac_end) of the blocks in 1/65536 units (host-weighted split).
template <typename RegionT, typename CountsT, typename TableT>
inline CountAwareRange count_aware_block_range_frac(
    Noc& noc,
    const RegionT& region,
    const CountsT& counts,
    const TableT& table,
    CircularBuffer& region_scratch,
    CircularBuffer& counts_scratch,
    CircularBuffer& table_scratch,
    uint32_t experts_per_chip,
    uint32_t max_rows,
    uint32_t tile_h,
    uint32_t chunks_per_row,
    uint32_t frac_start,
    uint32_t frac_end) {
    const uint32_t valid_tile_rows = count_aware_valid_tile_rows(
        noc, region, counts, table, region_scratch, counts_scratch, table_scratch, experts_per_chip, max_rows, tile_h);
    const uint64_t total_blocks = static_cast<uint64_t>(valid_tile_rows) * chunks_per_row;
    const uint32_t start = static_cast<uint32_t>((total_blocks * frac_start) >> 16);
    const uint32_t end = static_cast<uint32_t>((total_blocks * frac_end) >> 16);
    return {start, end - start};
}

template <typename RegionT, typename CountsT, typename TableT>
inline CountAwareRange count_aware_block_range(
    Noc& noc,
    const RegionT& region,
    const CountsT& counts,
    const TableT& table,
    CircularBuffer& region_scratch,
    CircularBuffer& counts_scratch,
    CircularBuffer& table_scratch,
    uint32_t experts_per_chip,
    uint32_t max_rows,
    uint32_t tile_h,
    uint32_t chunks_per_row,
    uint32_t core_id,
    uint32_t num_cores) {
    const uint32_t valid_tile_rows = count_aware_valid_tile_rows(
        noc, region, counts, table, region_scratch, counts_scratch, table_scratch, experts_per_chip, max_rows, tile_h);
    const uint32_t total_blocks = valid_tile_rows * chunks_per_row;
    const uint32_t start = (total_blocks * core_id) / num_cores;
    const uint32_t end = (total_blocks * (core_id + 1)) / num_cores;
    return {start, end - start};
}
