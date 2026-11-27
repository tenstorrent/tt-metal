// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat writer (BRISC, NOC1), shared by both regimes. Stores each block of `block_pages`
// consecutive output pages from cb_output_pages: one whole-page write per page, or only the valid
// rows when the page's tile row has v <= trim_rows_max rows (the output's padding rows may hold
// anything). Raw dataflow API: no kernel_lib helper writes pages to an interleaved buffer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_concat_tile_rows.hpp"

constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
constexpr uint32_t trim_rows_max = get_compile_time_arg_val(3);
constexpr auto output_args = TensorAccessorArgs<4>();

static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "every transfer is issued as a single NoC packet");

// Common runtime args: [out_addr, block_pages, out_Wt, out_Ht, v_last]
constexpr uint32_t CARG_OUT_ADDR = 0;
constexpr uint32_t CARG_BLOCK_PAGES = 1;
constexpr uint32_t CARG_OUT_WT = 2;
constexpr uint32_t CARG_OUT_HT = 3;
constexpr uint32_t CARG_LAST_VALID_ROWS = 4;

template <typename Accessor>
void store_block(uint32_t extent, uint32_t page, g6c::OutputRowCursor& rows, uint32_t slot, const Accessor& acc) {
    for (uint32_t i = 0; i < extent; ++i) {
        const uint32_t valid_rows = rows.valid_rows();
        if (g6c::trimmed<trim_rows_max>(valid_rows)) {
            g6c::for_each_valid_row_span(valid_rows, row_bytes, [&](uint32_t offset, uint32_t bytes) {
                noc_async_write<NOC_MAX_BURST_SIZE>(slot + offset, acc.get_noc_addr(page, offset), bytes);
            });
        } else {
            noc_async_write<NOC_MAX_BURST_SIZE>(slot, acc.get_noc_addr(page), page_bytes);
        }
        rows.advance();
        ++page;
        slot += page_bytes;
    }
    // The slots may be refilled once the data has left L1; completion is awaited once at the end.
    noc_async_writes_flushed();
}

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(CARG_OUT_ADDR);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(CARG_BLOCK_PAGES);

    // Per-core runtime args: [pages, first output page, its tile_row, its col]
    const uint32_t pages_this_core = get_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(1);
    g6c::OutputRowCursor rows{
        .tile_row = get_arg_val<uint32_t>(2),
        .col = get_arg_val<uint32_t>(3),
        .tiles_per_row = get_common_arg_val<uint32_t>(CARG_OUT_WT),
        .tile_rows = get_common_arg_val<uint32_t>(CARG_OUT_HT),
        .last_valid_rows = get_common_arg_val<uint32_t>(CARG_LAST_VALID_ROWS)};

    const auto acc = TensorAccessor(output_args, out_addr, page_bytes);

    uint32_t remaining = pages_this_core;
    while (remaining > 0) {
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6C-WAIT");
            cb_wait_front(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-STORE");
            store_block(extent, page, rows, get_read_ptr(cb_output_pages), acc);
        }
        cb_pop_front(cb_output_pages, block_pages);
        page += extent;
        remaining -= extent;
    }
    {
        DeviceZoneScopedN("G6C-DRAIN");
        noc_async_write_barrier();
    }
}
