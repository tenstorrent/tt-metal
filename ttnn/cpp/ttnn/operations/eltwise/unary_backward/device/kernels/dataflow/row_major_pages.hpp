// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Row-major addressing shared by the unary-backward row-major reader and writer.
//
// A row-major tensor is paged by row, but a page is not always a whole row: a width- or
// block-sharded row-major buffer pages each row by shard width, so one row of row_elements
// elements spans pages_per_row pages of page_elements each (the last possibly partial, for an
// uneven width shard). The logical page of element c in row r is r * pages_per_row +
// c / page_elements. Operands may be paged differently -- one interleaved, another width-sharded
// -- so every operand carries its own page geometry, and a CB page is assembled from however many
// page pieces that operand needs. The CB side is laid out by element, independent of paging, so
// all operands and the output line up element for element.
//
// NoC ALIGNMENT. A shard width need not be a multiple of the NoC alignment (a 100-wide bf16 row
// split over two cores is two 100-byte pages), so a piece can start or end at an address the NoC
// cannot transfer to or from exactly -- measured as silently wrong results, not a fault. A piece
// is moved directly only when both ends are aligned and its tail cannot spill onto a neighbouring
// piece; any other piece is staged through an aligned scratch window and placed with a core-local
// copy. Pages are allocated at aligned_page_size stride, so an aligned window around a piece never
// leaves the piece's own page.
struct RowMajorPaging {
    uint32_t page_elements;
    uint32_t pages_per_row;
    uint32_t element_bytes;
    uint32_t alignment;  // NoC alignment of the tensor's buffer (L1 or DRAM)

    uint32_t page_bytes() const { return page_elements * element_bytes; }
};

// Calls fn(page_id, page_offset_bytes, cb_offset_bytes, bytes) for each page piece covering
// elements [first, last) of `row`, placed in the CB page at element cb_base.
template <typename Fn>
inline void for_each_row_piece(
    const RowMajorPaging& paging, uint32_t row, uint32_t first, uint32_t last, uint32_t cb_base, Fn&& fn) {
    uint32_t page = first / paging.page_elements;
    uint32_t offset = first - (page * paging.page_elements);
    for (uint32_t c = first; c < last;) {
        const uint32_t room = paging.page_elements - offset;
        const uint32_t n = (last - c < room) ? (last - c) : room;
        fn(row * paging.pages_per_row + page,
           offset * paging.element_bytes,
           (cb_base + (c - first)) * paging.element_bytes,
           n * paging.element_bytes);
        c += n;
        ++page;
        offset = 0;
    }
}

inline uint32_t round_up_to(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

inline void copy_bytes(uint32_t dst, uint32_t src, uint32_t bytes) {
    auto* d = reinterpret_cast<volatile uint8_t*>(dst);
    const auto* s = reinterpret_cast<const volatile uint8_t*>(src);
    for (uint32_t i = 0; i < bytes; ++i) {
        d[i] = s[i];
    }
}

// Reads `bytes` from `page` at page_offset into L1 at cb_addr. `region_end` is the L1 address up
// to which the destination may be overwritten (the end of this row's slot in the CB page).
// `scratch` is an L1 buffer of at least round_up(bytes + 2 * alignment) bytes, itself aligned.
template <typename Accessor>
inline void read_piece(
    const Accessor& src,
    const RowMajorPaging& paging,
    uint32_t page,
    uint32_t page_offset,
    uint32_t cb_addr,
    uint32_t bytes,
    uint32_t region_end,
    uint32_t scratch) {
    const uint32_t a = paging.alignment;
    const bool aligned_ends = (page_offset % a) == 0 && (cb_addr % a) == 0;
    const bool tail_safe = (bytes % a) == 0 || cb_addr + round_up_to(bytes, a) <= region_end;
    if (aligned_ends && tail_safe) {
        noc_async_read(src.get_noc_addr(page, page_offset), cb_addr, bytes);
        return;
    }
    const uint32_t window_start = (page_offset / a) * a;
    const uint32_t delta = page_offset - window_start;
    noc_async_read(src.get_noc_addr(page, window_start), scratch, round_up_to(delta + bytes, a));
    noc_async_read_barrier();
    copy_bytes(cb_addr, scratch + delta, bytes);
}

// Writes `bytes` from L1 at cb_addr into `page` at page_offset. Unaligned pieces read-modify-write
// an aligned window of the page; every page of a row is written by the core that owns the row, so
// the window's neighbouring bytes are not being written by anyone else concurrently.
template <typename Accessor>
inline void write_piece(
    const Accessor& dst,
    const RowMajorPaging& paging,
    uint32_t page,
    uint32_t page_offset,
    uint32_t cb_addr,
    uint32_t bytes,
    uint32_t scratch) {
    const uint32_t a = paging.alignment;
    const bool aligned_ends = (page_offset % a) == 0 && (cb_addr % a) == 0;
    const bool tail_safe = (bytes % a) == 0 || page_offset + bytes == paging.page_bytes();
    if (aligned_ends && tail_safe) {
        noc_async_write(cb_addr, dst.get_noc_addr(page, page_offset), bytes);
        return;
    }
    // Earlier pieces of this page may still be in flight, and the window may cover their bytes.
    noc_async_write_barrier();
    const uint32_t window_start = (page_offset / a) * a;
    const uint32_t delta = page_offset - window_start;
    const uint32_t window_bytes = round_up_to(delta + bytes, a);
    const uint64_t window = dst.get_noc_addr(page, window_start);
    noc_async_read(window, scratch, window_bytes);
    noc_async_read_barrier();
    copy_bytes(scratch + delta, cb_addr, bytes);
    noc_async_write(scratch, window, window_bytes);
    noc_async_write_barrier();
}

// The scratch CB is sized with slack so its start can be rounded up to the widest alignment.
inline uint32_t aligned_scratch(uint32_t cb_id) { return round_up_to(get_write_ptr(cb_id), 64); }
