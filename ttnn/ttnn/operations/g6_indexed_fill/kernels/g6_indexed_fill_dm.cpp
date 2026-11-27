// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_indexed_fill: the op's only kernel (one data-movement kernel per core, no compute kernel).
//
// A core owns the contiguous work-unit range [unit_begin, unit_begin + unit_count). Unit
// u = page_idx * row_chunks_per_page + c names chunk c of page page_idx, where pages
// [0, P) are input_tensor pages and pages [P, P + K*R) are value_tensor pages. Every source chunk is
// loaded unconditionally (the loads need no index, so they are issued in the shadow of the ids
// read), then select_block decides each unit's destination page from batch_ids, then store_block
// writes every kept unit exactly once:
//   x unit, batch b = page_idx / R : SKIP if any ids[j] == b, else page page_idx
//   value unit, entry i, row r      : SKIP if ids[i] >= B or a later ids[j] == ids[i], else page ids[i]*R + r
// so every result page-chunk has exactly one writer across the grid and duplicates resolve to the
// largest i. batch_ids is read on the device in every run; no position is ever an argument.
//
// Helper library: none applies. There is no compute phase, and the dataflow helpers move tiles or
// self-aimed L1 copies; these are DRAM row chunks addressed by page id and byte offset. All CBs are
// plain L1 scratch of this one kernel (no reserve/push/wait/pop): ordering is by NoC barriers.

#include <cstdint>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t SKIP = 0xFFFFFFFFu;

constexpr uint32_t align_down(uint32_t v, uint32_t a) { return v - (v % a); }
constexpr uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
constexpr uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_batch_ids = get_compile_time_arg_val(0);
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(1);
    constexpr uint32_t cb_block_dest = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t P = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t row_chunk_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t row_chunks_per_page = get_compile_time_arg_val(9);
    constexpr uint32_t block_units = get_compile_time_arg_val(10);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(11);
    constexpr uint32_t ids_chunk_entries = get_compile_time_arg_val(12);
    constexpr uint32_t ids_chunks = get_compile_time_arg_val(13);
    constexpr bool ids_resident = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t dram_align = get_compile_time_arg_val(15);

    constexpr auto ids_args = TensorAccessorArgs<16>();
    constexpr auto x_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    // x units precede value units on the unit axis.
    constexpr uint32_t x_units_end = P * row_chunks_per_page;
    constexpr uint32_t last_chunk = row_chunks_per_page - 1;
    constexpr uint32_t last_chunk_bytes = page_bytes - last_chunk * row_chunk_bytes;
    constexpr uint32_t slot_set_bytes = block_units * row_chunk_bytes;

    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t y_addr = get_arg_val<uint32_t>(3);
    const uint32_t unit_begin = get_arg_val<uint32_t>(4);
    const uint32_t unit_count = get_arg_val<uint32_t>(5);
    if (unit_count == 0) {
        return;
    }

    const auto ids_acc = TensorAccessor(ids_args, ids_addr);
    const auto x_acc = TensorAccessor(x_args, x_addr);
    const auto s_acc = TensorAccessor(s_args, s_addr);
    const auto y_acc = TensorAccessor(y_args, y_addr);

    const uint32_t ids_l1 = get_write_ptr(cb_batch_ids);
    const uint32_t staged_l1 = get_write_ptr(cb_staged_pages);
    tt_l1_ptr uint32_t* dest = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_block_dest));

    const uint32_t num_blocks = div_up(unit_count, block_units);

    // Entries [j0, j1) are read as the DRAM-aligned superset [align_down(4*j0), align(4*j1)) of
    // the ids page; returns the L1 address of entry j0.
    auto issue_ids_window = [&](uint32_t window) -> uint32_t {
        const uint32_t j0 = window * ids_chunk_entries;
        const uint32_t j1 = min_u32(K, j0 + ids_chunk_entries);
        const uint32_t start = align_down(4 * j0, dram_align);
        const uint32_t end = align(4 * j1, dram_align);
        noc_async_read(ids_acc.get_noc_addr(0, start), ids_l1, end - start);
        return ids_l1 + (4 * j0 - start);
    };

    auto block_first = [&](uint32_t block_idx) { return unit_begin + block_idx * block_units; };
    auto block_size = [&](uint32_t block_idx) { return min_u32(block_units, unit_count - block_idx * block_units); };
    auto slot_set = [&](uint32_t block_idx) { return staged_l1 + (block_idx % staging_depth) * slot_set_bytes; };

    auto load_block = [&](uint32_t block_idx) {
        DeviceZoneScopedN("IF-LOAD-ISSUE");
        const uint32_t first = block_first(block_idx);
        const uint32_t n = block_size(block_idx);
        uint32_t slot = slot_set(block_idx);
        uint32_t page = first / row_chunks_per_page;
        uint32_t c = first - page * row_chunks_per_page;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t offset = c * row_chunk_bytes;
            const uint32_t read_bytes = align(c == last_chunk ? last_chunk_bytes : row_chunk_bytes, dram_align);
            const uint64_t src = page < P ? x_acc.get_noc_addr(page, offset) : s_acc.get_noc_addr(page - P, offset);
            noc_async_read(src, slot, read_bytes);
            slot += row_chunk_bytes;
            if (++c == row_chunks_per_page) {
                c = 0;
                ++page;
            }
        }
    };

    // Writes one destination page (or SKIP) per unit of the block into cb_block_dest.
    auto select_block = [&](uint32_t block_idx, uint32_t resident_window_l1) {
        DeviceZoneScopedN("IF-SELECT");
        const uint32_t first = block_first(block_idx);
        const uint32_t end = first + block_size(block_idx);
        const uint32_t x_end = min_u32(end, x_units_end);
        const uint32_t v_first = max_u32(first, x_units_end);
        const bool has_x = first < x_end;
        const bool has_v = v_first < end;

        // x units start as "keep at the same page"; value units start unassigned (SKIP) and take
        // their destination when the window holding their entry is scanned.
        {
            uint32_t page = first / row_chunks_per_page;
            uint32_t c = first - page * row_chunks_per_page;
            for (uint32_t u = first; u < end; ++u) {
                dest[u - first] = u < x_units_end ? page : SKIP;
                if (++c == row_chunks_per_page) {
                    c = 0;
                    ++page;
                }
            }
        }

        const uint32_t b_lo = has_x ? (first / row_chunks_per_page) / R : 0;
        const uint32_t b_hi = has_x ? ((x_end - 1) / row_chunks_per_page) / R : 0;
        const uint32_t i_lo = has_v ? (v_first / row_chunks_per_page - P) / R : 0;
        const uint32_t i_hi = has_v ? ((end - 1) / row_chunks_per_page - P) / R : 0;

        for (uint32_t window = 0; window < ids_chunks; ++window) {
            uint32_t entries_l1 = resident_window_l1;
            if constexpr (!ids_resident) {
                entries_l1 = issue_ids_window(window);
                DeviceZoneScopedN("IF-IDS-WAIT");
                noc_async_read_barrier();
            }
            volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(entries_l1);
            const uint32_t j0 = window * ids_chunk_entries;
            const uint32_t j1 = min_u32(K, j0 + ids_chunk_entries);

            if (has_x) {
                for (uint32_t j = j0; j < j1; ++j) {
                    const uint32_t t = ids[j - j0];
                    if (t < b_lo || t > b_hi) {
                        continue;
                    }
                    const uint32_t u_lo = max_u32(t * R * row_chunks_per_page, first);
                    const uint32_t u_hi = min_u32((t + 1) * R * row_chunks_per_page, x_end);
                    for (uint32_t u = u_lo; u < u_hi; ++u) {
                        dest[u - first] = SKIP;
                    }
                }
            }

            if (has_v) {
                for (uint32_t i = i_lo; i <= i_hi && i < j1; ++i) {
                    const uint32_t u_lo = max_u32((P + i * R) * row_chunks_per_page, v_first);
                    const uint32_t u_hi = min_u32((P + (i + 1) * R) * row_chunks_per_page, end);
                    uint32_t t;
                    uint32_t later_from;
                    if (i < j0) {
                        // Assigned in an earlier window; a SKIP is final.
                        const uint32_t d = dest[u_lo - first];
                        if (d == SKIP) {
                            continue;
                        }
                        t = d / R;
                        later_from = j0;
                    } else {
                        t = ids[i - j0];
                        if (t >= B) {
                            continue;
                        }
                        later_from = i + 1;
                        uint32_t page = u_lo / row_chunks_per_page;
                        uint32_t c = u_lo - page * row_chunks_per_page;
                        const uint32_t dest_base = t * R - (P + i * R);
                        for (uint32_t u = u_lo; u < u_hi; ++u) {
                            dest[u - first] = dest_base + page;
                            if (++c == row_chunks_per_page) {
                                c = 0;
                                ++page;
                            }
                        }
                    }
                    for (uint32_t j = later_from; j < j1; ++j) {
                        if (ids[j - j0] == t) {
                            for (uint32_t u = u_lo; u < u_hi; ++u) {
                                dest[u - first] = SKIP;
                            }
                            break;
                        }
                    }
                }
            }
        }
    };

    auto store_block = [&](uint32_t block_idx) {
        DeviceZoneScopedN("IF-STORE-ISSUE");
        const uint32_t first = block_first(block_idx);
        const uint32_t n = block_size(block_idx);
        uint32_t slot = slot_set(block_idx);
        uint32_t c = first % row_chunks_per_page;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t d = dest[k];
            if (d != SKIP) {
                const uint32_t offset = c * row_chunk_bytes;
                noc_async_write(
                    slot, y_acc.get_noc_addr(d, offset), c == last_chunk ? last_chunk_bytes : row_chunk_bytes);
            }
            slot += row_chunk_bytes;
            if (++c == row_chunks_per_page) {
                c = 0;
            }
        }
    };

    auto refill = [&](uint32_t block_idx) {
        if (block_idx >= num_blocks) {
            return;
        }
        if (block_idx >= staging_depth) {
            // The slot set still holds block (block_idx - staging_depth), whose writes were issued.
            DeviceZoneScopedN("IF-SLOT-WAIT");
            noc_async_writes_flushed();
        }
        load_block(block_idx);
    };

    uint32_t resident_entries_l1 = 0;
    if constexpr (ids_resident) {
        DeviceZoneScopedN("IF-IDS-ISSUE");
        resident_entries_l1 = issue_ids_window(0);
    }
    load_block(0);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        {
            DeviceZoneScopedN("IF-READ-WAIT");
            noc_async_read_barrier();
        }
        select_block(block_idx, resident_entries_l1);
        if constexpr (staging_depth > 1) {
            refill(block_idx + 1);
        }
        store_block(block_idx);
        if constexpr (staging_depth == 1) {
            refill(block_idx + 1);
        }
    }

    {
        DeviceZoneScopedN("IF-WRITE-WAIT");
        noc_async_write_barrier();
    }
}
