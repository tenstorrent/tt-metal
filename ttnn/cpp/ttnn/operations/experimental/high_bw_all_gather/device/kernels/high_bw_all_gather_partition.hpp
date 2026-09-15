// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Page-partition closed form shared by the HOST factory and the DEVICE kernels.
//
// Why it exists: `gathered_dim_size` (the active gathered extent) is hash-excluded, so one cached
// program serves every prefix length and override_runtime_arguments() re-derives every dependent
// per-worker runtime argument at dispatch. A ttnn trace REPLAY never runs that host patch, so a captured
// program would replay the captured chunk's page partition -- which for chunked prefill means gathering
// only the first chunk's prefix on every later chunk, leaving the rest of the KV unread.
//
// The trace-safe path therefore derives the same values ON-DEVICE. That only stays correct if the reader
// and the writer agree with each other AND with the host to the page: they share a semaphore protocol
// whose counts come out of this arithmetic, so a mismatch does not corrupt data, it HANGS. Hence one
// implementation here that all three include, rather than three transcriptions of the same formula.
//
// Dependency-free by construction (only <cstdint>, and a local min): it is included by kernels, which
// cannot see tt-metalium headers or reliably see <algorithm>.

#include <cstdint>

namespace ttnn::operations::experimental::high_bw_all_gather::partition {

inline constexpr uint32_t part_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline constexpr uint32_t part_max(uint32_t a, uint32_t b) { return a > b ? a : b; }

// Active source pages for a given gathered extent, counted in whole block-cyclic SLABS.
//
// Counting slabs rather than rows is what makes this layout-agnostic. A ROW_MAJOR page covers one
// gathered row, so pages-per-row is an integer; a TILE page covers 32 rows, so pages-per-row is
// fractional and a row-based multiply cannot be exact. Every extent this op is ever asked for is a whole
// number of slabs (gathered_dim_size_for_prefix rounds up to one), so `pages_per_slab` -- the worst-case
// page count divided by the slab count -- is an integer for both layouts and the multiply stays exact.
inline constexpr uint32_t active_num_input_pages(
    uint32_t gathered_dim_size, uint32_t slab_global, uint32_t pages_per_slab) {
    if (slab_global == 0) {
        return 0;
    }
    return (gathered_dim_size / slab_global) * pages_per_slab;
}

// Round a populated prefix up to whole block-cyclic slabs and clamp to the full extent. This is the
// caller-side derivation of `gathered_dim_size` itself: block-cyclic storage is only meaningful in
// complete slabs, and the gather must never claim more than the allocation.
inline constexpr uint32_t gathered_dim_size_for_prefix(
    uint32_t populated_global, uint32_t slab_global, uint32_t full_gathered_dim_size) {
    if (slab_global == 0) {
        return full_gathered_dim_size;
    }
    const uint32_t rounded = ((populated_global + slab_global - 1) / slab_global) * slab_global;
    return part_min(rounded, full_gathered_dim_size);
}

// One worker's contiguous page range under the plain (non-bank-owned) split: pages are divided evenly
// across `total_slices` and the first `remainder` slices take one extra.
struct WorkerPageRange {
    uint32_t input_page_start;
    uint32_t input_page_end;
    uint32_t page_count;
};
inline constexpr WorkerPageRange even_worker_page_range(
    uint32_t num_input_pages, uint32_t total_slices, uint32_t slice_idx) {
    if (total_slices == 0) {
        return {0, 0, 0};
    }
    const uint32_t per_slice = num_input_pages / total_slices;
    const uint32_t remainder = num_input_pages % total_slices;
    const uint32_t start = slice_idx * per_slice + part_min(slice_idx, remainder);
    const uint32_t end = (slice_idx + 1) * per_slice + part_min(slice_idx + 1, remainder);
    return {start, end, end - start};
}

// data_valid semaphore granularity in CB pages. Must match on both sides of the protocol: the reader
// waits on it and the writer signals in these units.
inline constexpr uint32_t data_valid_granularity_pages(
    uint32_t input_page_size,
    uint32_t output_chunk_size,
    uint32_t num_output_chunks,
    uint32_t packet_size,
    uint32_t total_slices) {
    if (input_page_size == 0 || output_chunk_size == 0 || total_slices == 0) {
        return 1;
    }
    const uint32_t pages_per_packet = part_max(1u, packet_size / input_page_size);
    const uint32_t cb_page_size = input_page_size * pages_per_packet;
    const uint32_t outputs_per_cb_page = part_max(1u, cb_page_size / output_chunk_size);
    const uint32_t cb_pages_per_stripe = part_max(1u, (num_output_chunks / total_slices) / outputs_per_cb_page);
    return part_max(1u, cb_pages_per_stripe / 2u);
}

// One worker's page range under the OUTPUT-BANK-OWNED schedule. Moved here from the host-only scheduler
// header so the kernels can evaluate the identical mapping; scheduler::derive_bank_owned_slice delegates.
// Assumes interleaved DRAM page p lives in bank p % num_dram_banks at bank-local offset p / num_dram_banks.
struct BankOwnedPageRange {
    uint32_t bank;
    uint32_t input_page_start;
    uint32_t page_count;
    bool valid;
};
inline constexpr BankOwnedPageRange bank_owned_page_range(
    uint32_t num_input_pages,
    uint32_t num_links,
    uint32_t workers_per_direction,
    uint32_t num_dram_banks,
    uint32_t link,
    uint32_t worker) {
    if (num_links == 0 || workers_per_direction == 0 || num_dram_banks == 0 || link >= num_links ||
        worker >= workers_per_direction) {
        return {0, 0, 0, false};
    }
    const uint32_t total_workers = num_links * workers_per_direction;
    if (total_workers < num_dram_banks) {
        return {0, 0, 0, false};
    }
    uint32_t bank = 0;
    uint32_t worker_in_bank = 0;
    uint32_t workers_for_bank = 0;
    if (num_dram_banks % num_links == 0 && workers_per_direction % (num_dram_banks / num_links) == 0) {
        // Evenly divisible: each link owns interleaved banks and adjacent workers split a bank.
        const uint32_t banks_per_link = num_dram_banks / num_links;
        workers_for_bank = workers_per_direction / banks_per_link;
        bank = link + (worker / workers_for_bank) * num_links;
        worker_in_bank = worker % workers_for_bank;
    } else {
        // Interleave links in flat order so harvested configurations keep a similar link-to-bank pattern.
        const uint32_t global_worker = worker * num_links + link;
        bank = global_worker % num_dram_banks;
        worker_in_bank = global_worker / num_dram_banks;
        workers_for_bank = total_workers <= bank ? 0 : 1 + (total_workers - 1 - bank) / num_dram_banks;
    }
    if (workers_for_bank == 0) {
        return {bank, 0, 0, false};
    }
    const uint32_t bank_page_count = num_input_pages <= bank ? 0 : 1 + (num_input_pages - 1 - bank) / num_dram_banks;
    const uint32_t pages_per_worker = bank_page_count / workers_for_bank;
    const uint32_t remainder = bank_page_count % workers_for_bank;
    const uint32_t worker_page_offset = worker_in_bank * pages_per_worker + part_min(worker_in_bank, remainder);
    const uint32_t worker_page_count = pages_per_worker + (worker_in_bank < remainder ? 1u : 0u);
    return {bank, bank + worker_page_offset * num_dram_banks, worker_page_count, true};
}

// Everything one (link, worker, direction) needs, derived from the active page count. The host computes
// it at program build and each kernel recomputes it from an on-device page count on the trace-safe path,
// so all three agree by construction rather than by three copies of the same formula.
struct WorkerSchedule {
    uint32_t input_page_start;
    uint32_t input_page_end;
    uint32_t local_output_start;
    uint32_t slice_count;
    uint32_t final_start;
    uint32_t final_count;
    uint32_t total_chunks;
    uint32_t data_valid_granularity;
};
inline constexpr WorkerSchedule worker_schedule(
    uint32_t num_input_pages,
    uint32_t split_factor,
    uint32_t total_slices,
    uint32_t slice_idx,
    bool bank_owned,
    uint32_t num_links,
    uint32_t workers_per_direction,
    uint32_t num_dram_banks,
    uint32_t link,
    uint32_t worker,
    uint32_t slice_step,
    bool ring_even_split,
    bool is_forward,
    uint32_t num_recv,
    uint32_t input_page_size,
    uint32_t output_chunk_size,
    uint32_t packet_size) {
    const uint32_t num_output_chunks = num_input_pages * split_factor;

    uint32_t input_page_start = 0;
    uint32_t page_count = 0;
    uint32_t input_page_end = 0;
    if (bank_owned) {
        const auto bank_range =
            bank_owned_page_range(num_input_pages, num_links, workers_per_direction, num_dram_banks, link, worker);
        input_page_start = bank_range.input_page_start;
        page_count = bank_range.page_count;
        // Bank-owned pages are strided by num_dram_banks, so the end is start + count * stride.
        input_page_end = input_page_start + page_count * num_dram_banks;
    } else {
        const auto even = even_worker_page_range(num_input_pages, total_slices, slice_idx);
        input_page_start = even.input_page_start;
        input_page_end = even.input_page_end;
        page_count = even.page_count;
    }

    // Output placement. num_output_chunks / num_input_pages == split_factor exactly, so the ratio the
    // host writes as (start * num_output_chunks) / num_input_pages is that product -- computed as a
    // product here to stay exact and to avoid a divide by a possibly-zero page count.
    const uint32_t local_output_start = bank_owned ? input_page_start : input_page_start * split_factor;
    const uint32_t local_output_end = bank_owned ? local_output_start + page_count : input_page_end * split_factor;
    const uint32_t slice_count = local_output_end - local_output_start;

    const uint32_t half = slice_count / 2;
    const uint32_t final_start = ring_even_split
                                     ? (is_forward ? local_output_start : local_output_start + half * slice_step)
                                     : local_output_start;
    const uint32_t final_count = ring_even_split ? (is_forward ? half : slice_count - half) : slice_count;
    const uint32_t total_chunks = num_recv * slice_count - (ring_even_split ? slice_count - final_count : 0);

    return {
        input_page_start,
        input_page_end,
        local_output_start,
        slice_count,
        final_start,
        final_count,
        total_chunks,
        data_valid_granularity_pages(input_page_size, output_chunk_size, num_output_chunks, packet_size, total_slices),
    };
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather::partition
