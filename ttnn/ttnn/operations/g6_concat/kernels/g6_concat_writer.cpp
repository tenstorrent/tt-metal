// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared writer of every regime (BRISC, NOC1): stores this core's contiguous output page range (tiles,
// or sticks for ROW_MAJOR) from cb_output_pages, a block of block_pages at a time.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_concat_tile_rows.hpp"

namespace {

// Common runtime args; layout mirrors _writer_common_args() / WRITER_COMMON_OUTPUT_BASE in
// g6_concat_program_descriptor.py.
constexpr uint32_t CRTA_OUTPUT_BASE = 0;
constexpr uint32_t CRTA_BLOCK_PAGES = 1;
constexpr uint32_t CRTA_OUT_WT = 2;
constexpr uint32_t CRTA_OUT_HT = 3;
constexpr uint32_t CRTA_LAST_ROW_VALID_ROWS = 4;

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(CRTA_OUTPUT_BASE), page_bytes);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(CRTA_BLOCK_PAGES);
    g6_concat::OutputRowCursor rows(
        start_page,
        get_common_arg_val<uint32_t>(CRTA_OUT_WT),
        get_common_arg_val<uint32_t>(CRTA_OUT_HT),
        get_common_arg_val<uint32_t>(CRTA_LAST_ROW_VALID_ROWS));

    uint32_t output_page = start_page;
    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const bool last_block = block_idx + 1 == num_blocks_this_core;
        const uint32_t block_extent = last_block ? core_pages - block_idx * block_pages : block_pages;

        {
            DeviceZoneScopedN("G6C-WAIT");
            cb_wait_front(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-STORE");
            // store_block
            uint32_t slot_addr = get_read_ptr(cb_output_pages);
            for (uint32_t page = 0; page < block_extent; ++page) {
                const uint32_t valid_rows = rows.valid_rows();
                if (g6_concat::is_trimmed<trim_rows_max>(valid_rows)) {
                    g6_concat::for_each_valid_row_piece<face_row_bytes>(
                        valid_rows, [&](uint32_t offset_bytes, uint32_t size_bytes) {
                            noc_async_write_page(
                                output_page, output, slot_addr + offset_bytes, size_bytes, offset_bytes);
                        });
                } else {
                    noc_async_write_page(output_page, output, slot_addr);
                }
                slot_addr += page_bytes;
                ++output_page;
                rows.advance();
            }
            noc_async_writes_flushed();
        }
        // Nominal count on the ragged last block keeps the ring cycle closed.
        cb_pop_front(cb_output_pages, block_pages);
    }
    {
        DeviceZoneScopedN("G6C-DRAIN");
        noc_async_write_barrier();
    }
}
