// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC, NoC1): per block of output pages
//   element_shift only: assemble_columns_block — cb_staged_tiles -> cb_assembled_tiles
//   store_block — staged (page_copy, row_shift) or assembled (element_shift) pages -> out
// One write flush per block; one write barrier at exit (BB-03).

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"
#include "g6_slice_common.hpp"

using namespace g6_slice;

namespace {

// Output rows [0, rows) of out page `page`: row y is row y + row_offset of `top`, or of `bottom`
// once that passes the tile edge. One write per face-row run and column half; runs break where
// the destination or the source crosses a face (at most 4 runs -> 8 writes). Padding rows are not
// written.
template <typename Accessor>
FORCE_INLINE void store_shifted_rows(
    const Accessor& out, const Geometry& g, uint32_t page, uint32_t rows, uint32_t top, uint32_t bottom) {
    uint32_t y0 = 0;
    while (y0 < rows) {
        const uint32_t s0 = y0 + g.row_offset;
        const uint32_t dst_face_end = y0 < FACE ? FACE : TILE;
        const uint32_t src_face_end = (s0 / FACE + 1) * FACE - g.row_offset;
        const uint32_t y1 = min_u32(rows, min_u32(dst_face_end, src_face_end));
        const uint32_t src = s0 < TILE ? top : bottom;
        const uint32_t src_row = s0 % TILE;
        const uint32_t bytes = (y1 - y0) * face_row_bytes;
        noc_async_write_page(page, out, src + row_byte(src_row, 0), bytes, row_byte(y0, 0));
        noc_async_write_page(page, out, src + row_byte(src_row, 1), bytes, row_byte(y0, 1));
        y0 = y1;
    }
}

// Output columns [0, cols) of `dst`: column x is column x + col_offset of `left`, or of `right`
// once that passes the tile edge. All 32 rows of each column move.
FORCE_INLINE void assemble_columns(const Geometry& g, uint32_t cols, uint32_t left, uint32_t right, uint32_t dst) {
    using elem_t = std::conditional_t<element_bytes == 4, uint32_t, uint16_t>;
    const uint32_t right_start = TILE - g.col_offset;
    const uint32_t from_left = min_u32(cols, right_start);
    for (uint32_t x = 0; x < from_left; ++x) {
        dataflow_kernel_lib::copy_tile_column<cb_staged_tiles, elem_t>(left, x + g.col_offset, dst, x);
    }
    for (uint32_t x = from_left; x < cols; ++x) {
        dataflow_kernel_lib::copy_tile_column<cb_staged_tiles, elem_t>(right, x - right_start, dst, x);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<CT_ACCESSOR_OFFSET>();
    constexpr uint32_t slot_stride = sources_per_output * page_bytes;
    constexpr uint32_t assembled_block_pages = stencil_rows * block_pages;
    constexpr uint32_t assembled_stride = stencil_rows * page_bytes;

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const Geometry g = Geometry::load();
    const auto out = TensorAccessor(out_args, g.tensor_addr, page_bytes);

    OutPageWalker walker(g, start_page);
    for (uint32_t done = 0; done < num_pages; done += block_pages) {
        const uint32_t n = min_u32(block_pages, num_pages - done);
        const uint32_t first_page = start_page + done;
        {
            DeviceZoneScopedN("store_wait_staged");
            cb_wait_front(cb_staged_tiles, staged_block_pages);
        }
        const uint32_t staged = get_read_ptr(cb_staged_tiles);

        if constexpr (regime == REGIME_PAGE_COPY) {
            DeviceZoneScopedN("store_block");
            for (uint32_t q = 0; q < n; ++q) {
                noc_async_write_page(first_page + q, out, staged + q * page_bytes);
            }
        } else if constexpr (regime == REGIME_ROW_SHIFT) {
            DeviceZoneScopedN("store_block");
            for (uint32_t q = 0; q < n; ++q) {
                const uint32_t slot = staged + q * slot_stride;
                store_shifted_rows(out, g, first_page + q, g.valid_rows(walker.i), slot, slot + page_bytes);
                walker.advance();
            }
        } else {
            static_assert(element_bytes == 2 || element_bytes == 4, "column copy needs 2- or 4-byte elements");
            // The writer is this CB's only producer and consumer, so reserve/wait never block.
            cb_reserve_back(cb_assembled_tiles, assembled_block_pages);
            const uint32_t assembled = get_write_ptr(cb_assembled_tiles);
            {
                DeviceZoneScopedN("assemble_columns_block");
                OutPageWalker w = walker;
                for (uint32_t q = 0; q < n; ++q) {
                    const uint32_t slot = staged + q * slot_stride;
                    const uint32_t dst = assembled + q * assembled_stride;
                    const uint32_t cols = g.valid_cols(w.j);
                    assemble_columns(g, cols, slot, slot + page_bytes, dst);
                    if (g.needs_below(w.i)) {
                        const uint32_t bottom = slot + stencil_cols * page_bytes;
                        assemble_columns(g, cols, bottom, bottom + page_bytes, dst + page_bytes);
                    }
                    w.advance();
                }
            }
            cb_pop_front(cb_staged_tiles, staged_block_pages);
            cb_push_back(cb_assembled_tiles, assembled_block_pages);
            cb_wait_front(cb_assembled_tiles, assembled_block_pages);
            {
                DeviceZoneScopedN("store_block");
                for (uint32_t q = 0; q < n; ++q) {
                    const uint32_t top = assembled + q * assembled_stride;
                    if (g.row_offset == 0) {
                        noc_async_write_page(first_page + q, out, top);
                    } else {
                        store_shifted_rows(out, g, first_page + q, g.valid_rows(walker.i), top, top + page_bytes);
                    }
                    walker.advance();
                }
            }
        }

        {
            DeviceZoneScopedN("store_flush");
            noc_async_writes_flushed();
        }
        if constexpr (regime == REGIME_ELEMENT_SHIFT) {
            cb_pop_front(cb_assembled_tiles, assembled_block_pages);
        } else {
            cb_pop_front(cb_staged_tiles, staged_block_pages);
        }
    }
    {
        DeviceZoneScopedN("write_barrier");
        noc_async_write_barrier();
    }
}
