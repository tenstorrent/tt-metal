// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the g6_slice reader and writer: CT-arg layout, common-RT-arg layout, the output-page
// walker over the canonical (D0, D1, H, W) index space, and the in-tile byte geometry.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_slice {

constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;

// Mirrors REGIME_* in g6_slice_program_descriptor.py.
constexpr uint32_t REGIME_PAGE_COPY = 0;
constexpr uint32_t REGIME_ROW_SHIFT = 1;
constexpr uint32_t REGIME_ELEMENT_SHIFT = 2;

// CT args shared by both kernels; TensorAccessorArgs follow at CT_ACCESSOR_OFFSET.
constexpr uint32_t cb_staged_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_assembled_tiles = get_compile_time_arg_val(1);
constexpr uint32_t block_pages = get_compile_time_arg_val(2);
constexpr uint32_t regime = get_compile_time_arg_val(3);
constexpr uint32_t sources_per_output = get_compile_time_arg_val(4);
constexpr bool needs_right_any = get_compile_time_arg_val(5) != 0;
constexpr bool needs_below_any = get_compile_time_arg_val(6) != 0;
constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
constexpr uint32_t element_bytes = get_compile_time_arg_val(8);
constexpr uint32_t CT_ACCESSOR_OFFSET = 9;

constexpr uint32_t stencil_cols = needs_right_any ? 2 : 1;
constexpr uint32_t stencil_rows = needs_below_any ? 2 : 1;
static_assert(
    sources_per_output == (regime == REGIME_PAGE_COPY   ? 1
                           : regime == REGIME_ROW_SHIFT ? stencil_rows
                                                        : stencil_cols * stencil_rows),
    "sources_per_output disagrees with the regime's stencil");
static_assert(regime != REGIME_PAGE_COPY || (!needs_right_any && !needs_below_any));
static_assert(regime != REGIME_ROW_SHIFT || !needs_right_any);

// Staged-block quantum: every reserve/push/wait/pop moves the nominal block, ragged last block
// included, so a depth-2 ring wraps exactly at its end.
constexpr uint32_t staged_block_pages = sources_per_output * block_pages;

// Face-row and face bytes of one tile page.
constexpr uint32_t face_row_bytes = FACE * element_bytes;
constexpr uint32_t face_bytes = FACE * face_row_bytes;

// Byte offset of row `r`, column half `c`, inside one tile page.
FORCE_INLINE uint32_t row_byte(uint32_t r, uint32_t c) {
    return ((r >= FACE ? 2 : 0) + c) * face_bytes + (r % FACE) * face_row_bytes;
}

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Common RT args (COMMON_ARG_FIELDS in g6_slice_program_descriptor.py).
struct Geometry {
    uint32_t tensor_addr;
    uint32_t M_D1, out_h_tiles, out_w_tiles;
    uint32_t N_D1, in_h_tiles, in_w_tiles;
    uint32_t b_D0, b_D1, hb, wb;
    uint32_t row_offset, col_offset;
    uint32_t M_H, M_W;

    static Geometry load() {
        Geometry g;
        uint32_t i = 0;
        g.tensor_addr = get_common_arg_val<uint32_t>(i++);
        g.M_D1 = get_common_arg_val<uint32_t>(i++);
        g.out_h_tiles = get_common_arg_val<uint32_t>(i++);
        g.out_w_tiles = get_common_arg_val<uint32_t>(i++);
        g.N_D1 = get_common_arg_val<uint32_t>(i++);
        g.in_h_tiles = get_common_arg_val<uint32_t>(i++);
        g.in_w_tiles = get_common_arg_val<uint32_t>(i++);
        g.b_D0 = get_common_arg_val<uint32_t>(i++);
        g.b_D1 = get_common_arg_val<uint32_t>(i++);
        g.hb = get_common_arg_val<uint32_t>(i++);
        g.wb = get_common_arg_val<uint32_t>(i++);
        g.row_offset = get_common_arg_val<uint32_t>(i++);
        g.col_offset = get_common_arg_val<uint32_t>(i++);
        g.M_H = get_common_arg_val<uint32_t>(i++);
        g.M_W = get_common_arg_val<uint32_t>(i++);
        return g;
    }

    // Rows / columns of output tile (i, j) that hold logical output elements.
    uint32_t valid_rows(uint32_t i) const { return min_u32(TILE, M_H - TILE * i); }
    uint32_t valid_cols(uint32_t j) const { return min_u32(TILE, M_W - TILE * j); }
    bool needs_below(uint32_t i) const { return needs_below_any && row_offset + valid_rows(i) > TILE; }
    bool needs_right(uint32_t j) const { return needs_right_any && col_offset + valid_cols(j) > TILE; }
};

// Walks the flat output-page index row-major over (o0, o1, i, j) and tracks the source page A(p).
struct OutPageWalker {
    const Geometry& g;
    uint32_t o0, o1, i, j;
    uint32_t row_base;  // A of (o0, o1, i, j = 0)

    OutPageWalker(const Geometry& geometry, uint32_t page) : g(geometry) {
        j = page % g.out_w_tiles;
        page /= g.out_w_tiles;
        i = page % g.out_h_tiles;
        page /= g.out_h_tiles;
        o1 = page % g.M_D1;
        o0 = page / g.M_D1;
        rebase();
    }

    void rebase() {
        row_base = (((g.b_D0 + o0) * g.N_D1 + g.b_D1 + o1) * g.in_h_tiles + g.hb + i) * g.in_w_tiles + g.wb;
    }

    uint32_t source_page() const { return row_base + j; }

    void advance() {
        if (++j < g.out_w_tiles) {
            return;
        }
        j = 0;
        if (++i == g.out_h_tiles) {
            i = 0;
            if (++o1 == g.M_D1) {
                o1 = 0;
                ++o0;
            }
        }
        rebase();
    }
};

}  // namespace g6_slice
