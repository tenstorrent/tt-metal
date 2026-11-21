// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The walk of split, shared by the reader and the writer so that both derive the identical sequence of units
// from the same runtime args (op_design.md, Block schedule). The host derives every geometry field once
// (split_program_artifacts.py, derive_geometry) and passes them as common runtime args.

#pragma once

#include <cstdint>

// The regime is the define SPLIT_REGIME, set by split_program_artifacts.py (REGIME_IDS): a preprocessor value and
// not a compile-time arg, because only the shift-regime readers bind the scratchpad window.
#define SPLIT_REGIME_PAGE_REMAP 0
#define SPLIT_REGIME_TILE_ROW_SHIFT 1
#define SPLIT_REGIME_TILE_COL_SHIFT 2
#define SPLIT_REGIME_RM_SUBSTICK 3
// The three shift regimes share the window reader and the ShiftWalk below.
#define SPLIT_REGIME_IS_SHIFT (SPLIT_REGIME != SPLIT_REGIME_PAGE_REMAP)

namespace split {


inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// page_remap: a unit is one input page u = (o * cut_pages + p) * inner_pages + i, which is exactly one page of
// chunk c = p / chunk_pages, at page-row p - c * chunk_pages of that chunk.
struct PageRemapWalk {
    uint32_t cut_pages, inner_pages, chunk_pages;
    uint32_t o = 0, p = 0, i = 0;  // input coordinates
    uint32_t c = 0, pc = 0;        // chunk, page-row inside the chunk

    PageRemapWalk(uint32_t cut, uint32_t inner, uint32_t chunk) : cut_pages(cut), inner_pages(inner), chunk_pages(chunk) {}

    void seek(uint32_t u) {
        i = u % inner_pages;
        const uint32_t row = u / inner_pages;
        p = row % cut_pages;
        o = row / cut_pages;
        c = p / chunk_pages;
        pc = p - c * chunk_pages;
    }
    uint32_t input_page() const { return (o * cut_pages + p) * inner_pages + i; }
    uint32_t chunk() const { return c; }
    uint32_t chunk_extent() const { return min_u32(chunk_pages, cut_pages - c * chunk_pages); }
    uint32_t output_page() const { return (o * chunk_extent() + pc) * inner_pages + i; }
    void next() {
        if (++i < inner_pages) {
            return;
        }
        i = 0;
        if (++p == cut_pages) {
            p = 0, pc = 0, c = 0, ++o;
        } else if (++pc == chunk_pages) {
            pc = 0, ++c;
        }
    }
    void advance(uint32_t n) {
        for (uint32_t k = 0; k < n; ++k) {
            next();
        }
    }
};

// The shift regimes (tile_row_shift, tile_col_shift, rm_substick) share one walk. The input is viewed as
// [outer, cut_pages, inner_pages] pages; a "line" is one (outer, inner) pair, i.e. the cut_pages input pages a run of
// the cut axis crosses (line = o * inner_pages + i). A unit is one output page, walked line-major: for each line the
// output pages of chunk 0, then chunk 1, ... (line_units of them). Along the cut, an input page holds page_elems
// elements and an output page unit_elems; output page (line, c, j) takes its valid elements from input elements
// [first(), first() + valid()) of the line, which lie in input page ta() and, past its end, ta() + 1.
//
//   tile_row_shift  outer = images,             cut = tile rows (h_tiles), inner = w_tiles, page/unit = 32 rows
//   tile_col_shift  outer = images * h_tiles,   cut = tile cols (w_tiles), inner = 1,       page/unit = 32 cols
//   rm_substick     outer = sticks,             cut = 1 page (the stick),  inner = 1,       page = W, unit = split
struct ShiftWalk {
    uint32_t cut_pages, inner_pages, split_elems, cut_elems, unit_elems, page_elems, chunk_units, line_units;
    uint32_t line = 0;  // o * inner_pages + i
    uint32_t k = 0;     // position inside the line, = c * chunk_units + j
    uint32_t c = 0, j = 0;

    ShiftWalk(
        uint32_t cut, uint32_t inner, uint32_t split, uint32_t extent, uint32_t unit, uint32_t page, uint32_t cu,
        uint32_t lu) :
        cut_pages(cut),
        inner_pages(inner),
        split_elems(split),
        cut_elems(extent),
        unit_elems(unit),
        page_elems(page),
        chunk_units(cu),
        line_units(lu) {}

    void seek(uint32_t u) {
        line = u / line_units;
        k = u - line * line_units;
        c = k / chunk_units;
        j = k - c * chunk_units;
    }
    uint32_t chunk() const { return c; }
    uint32_t chunk_elems() const { return min_u32(split_elems, cut_elems - c * split_elems); }
    uint32_t chunk_pages() const { return (chunk_elems() + unit_elems - 1) / unit_elems; }
    uint32_t first() const { return c * split_elems + unit_elems * j; }
    uint32_t valid() const { return min_u32(unit_elems, chunk_elems() - unit_elems * j); }
    uint32_t shift() const { return first() % page_elems; }
    uint32_t ta() const { return first() / page_elems; }
    // Input page tb() == ta() or ta() + 1: the last input page the unit reads.
    uint32_t tb() const { return shift() + valid() > page_elems ? ta() + 1 : ta(); }
    // Input page at cut position t of this line.
    uint32_t input_page(uint32_t t) const {
        const uint32_t o = line / inner_pages, i = line - o * inner_pages;
        return (o * cut_pages + t) * inner_pages + i;
    }
    uint32_t output_page() const {
        const uint32_t o = line / inner_pages, i = line - o * inner_pages;
        return (o * chunk_pages() + j) * inner_pages + i;
    }
    void next() {
        if (++k == line_units) {
            k = 0, c = 0, j = 0, ++line;
        } else if (++j == chunk_units) {
            j = 0, ++c;
        }
    }
    void advance(uint32_t n) {
        for (uint32_t m = 0; m < n; ++m) {
            next();
        }
    }
};

}  // namespace split
