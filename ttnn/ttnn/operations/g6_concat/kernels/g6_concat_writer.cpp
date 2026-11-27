// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat writer (BRISC, NoC1): store_pages_block.
// Drains cb_moved_pages one block at a time and writes each output page's valid-row
// prefix to the interleaved output. Raw NoC + TensorAccessor: no kernel_lib dataflow
// helper copies whole or partial pages (op_design.md, Helpers considered and rejected).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_concat_pages.hpp"

void kernel_main() {
    constexpr uint32_t cb_moved_pages = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(3);
    constexpr bool cut_at_tile_rows = get_compile_time_arg_val(4) != 0;
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_tile_cols = get_common_arg_val<uint32_t>(1);
    const uint32_t out_tile_rows_per_image = get_common_arg_val<uint32_t>(2);
    const uint32_t out_rows_tail = get_common_arg_val<uint32_t>(3);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);
    g6c::OutPageCursor cursor(first_page, out_tile_cols, out_tile_rows_per_image, out_rows_tail);

    uint32_t pages_done = 0;
    while (pages_done < num_pages) {
        const uint32_t extent = g6c::block_extent<block_pages, cut_at_tile_rows>(num_pages - pages_done, cursor);
        {
            DeviceZoneScopedN("g6c_w_wait_front");
            cb_wait_front(cb_moved_pages, block_pages);
        }
        {
            DeviceZoneScopedN("g6c_store_pages_block");
            uint32_t slot_addr = get_read_ptr(cb_moved_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                const uint32_t page = cursor.page;
                g6c::for_each_valid_row_piece<page_bytes, face_row_bytes>(
                    cursor.valid_rows(), [&](uint32_t offset, uint32_t bytes) {
                        noc_async_write<page_bytes>(slot_addr + offset, out.get_noc_addr(page, offset), bytes);
                    });
                slot_addr += page_bytes;
                cursor.advance();
            }
        }
        {
            DeviceZoneScopedN("g6c_w_writes_flushed");
            noc_async_writes_flushed();
        }
        // Nominal pop: a ragged block narrows only the transfers, never the ring step.
        cb_pop_front(cb_moved_pages, block_pages);
        pages_done += extent;
    }
    {
        DeviceZoneScopedN("g6c_w_write_barrier");
        noc_async_write_barrier();
    }
}
