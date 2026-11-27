// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC, NoC0): stage_sources_block — x -> cb_staged_tiles, one block of
// output pages at a time, one read barrier per block. Slot layout of one output page:
// sources_per_output pages, below-major: [A, R][B, BR] (R / B present only when the stencil has them).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_slice_common.hpp"

using namespace g6_slice;

namespace {

// row_shift: one covering read of source rows [r0, r1] of `page`, landed at the source's own
// in-page offset so the DRAM read stays 64-B congruent (the shift is applied by the writer).
template <typename Accessor>
FORCE_INLINE void read_rows(const Accessor& x, uint32_t page, uint32_t r0, uint32_t r1, uint32_t slot) {
    const uint32_t lo = row_byte(r0, 0);
    const uint32_t hi = row_byte(r1, 1) + face_row_bytes;
    noc_async_read(x.get_noc_addr(page, lo), slot + lo, hi - lo);
}

template <typename Accessor>
FORCE_INLINE void stage_page(const Accessor& x, const Geometry& g, const OutPageWalker& w, uint32_t slot) {
    const uint32_t a = w.source_page();
    if constexpr (regime == REGIME_PAGE_COPY) {
        noc_async_read_page(a, x, slot);
    } else if constexpr (regime == REGIME_ROW_SHIFT) {
        const uint32_t last_row = g.row_offset + g.valid_rows(w.i) - 1;
        read_rows(x, a, g.row_offset, min_u32(last_row, TILE - 1), slot);
        if (g.needs_below(w.i)) {
            read_rows(x, a + g.in_w_tiles, 0, last_row - TILE, slot + page_bytes);
        }
    } else {
        const bool right = g.needs_right(w.j);
        noc_async_read_page(a, x, slot);
        if (right) {
            noc_async_read_page(a + 1, x, slot + page_bytes);
        }
        if (g.needs_below(w.i)) {
            const uint32_t bottom = slot + stencil_cols * page_bytes;
            noc_async_read_page(a + g.in_w_tiles, x, bottom);
            if (right) {
                noc_async_read_page(a + g.in_w_tiles + 1, x, bottom + page_bytes);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<CT_ACCESSOR_OFFSET>();

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const Geometry g = Geometry::load();
    const auto x = TensorAccessor(x_args, g.tensor_addr, page_bytes);

    OutPageWalker walker(g, start_page);
    for (uint32_t done = 0; done < num_pages; done += block_pages) {
        const uint32_t n = min_u32(block_pages, num_pages - done);
        {
            DeviceZoneScopedN("stage_wait_free");
            cb_reserve_back(cb_staged_tiles, staged_block_pages);
        }
        {
            DeviceZoneScopedN("stage_sources_block");
            uint32_t slot = get_write_ptr(cb_staged_tiles);
            for (uint32_t q = 0; q < n; ++q) {
                stage_page(x, g, walker, slot);
                walker.advance();
                slot += sources_per_output * page_bytes;
            }
        }
        {
            DeviceZoneScopedN("stage_read_barrier");
            noc_async_read_barrier();
            cb_push_back(cb_staged_tiles, staged_block_pages);
        }
    }
}
