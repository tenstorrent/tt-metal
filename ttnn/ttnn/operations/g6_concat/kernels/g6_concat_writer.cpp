// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat writer (BRISC / NOC1), shared by the page_copy and row_gather readers.
// store_block: one cb_wait_front / cb_pop_front of block_pages per block; per page one
// whole-page write, or only rows [0, v) when v <= TRIM_ROWS_MAX (the rest of that page is
// output padding, which the contract leaves unspecified).
//
// No kernel_lib helper covers a page store from a CB to an interleaved buffer (the eltwise /
// tilize helpers are compute-side and would round-trip values through DEST), so this kernel uses
// raw dataflow calls — see op_design.md, "Helpers considered and rejected".

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_concat_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(2);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();
    static_assert(trim_rows_max <= g6_concat::FACE_ROWS, "row trimming only covers faces 0 and 1");

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);
    g6_concat::OutputPageCursor out_cursor{
        .out_Wt = get_common_arg_val<uint32_t>(2),
        .out_Ht = get_common_arg_val<uint32_t>(3),
        .v_last = get_common_arg_val<uint32_t>(4),
        .R = 0,
        .c = 0};

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    out_cursor.init(start_page);

    const uint32_t num_blocks = (num_pages + block_pages - 1) / block_pages;
    uint32_t page = start_page;
    uint32_t remaining = num_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // The ragged last block transfers fewer pages but still waits/pops the nominal
        // block_pages so the CB ring closes on its fifo_limit.
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6C-WAIT");
            cb_wait_front(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-STORE");
            uint32_t slot = get_read_ptr(cb_output_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                const uint32_t v = out_cursor.valid_rows();
                if (v <= trim_rows_max) {
                    g6_concat::for_each_leading_rows_range(
                        v, face_row_bytes, face_bytes, [&](uint32_t offset, uint32_t bytes) {
                            noc_async_write(slot + offset, output.get_noc_addr(page, offset), bytes);
                        });
                } else {
                    noc_async_write(slot, output.get_noc_addr(page), page_bytes);
                }
                slot += page_bytes;
                ++page;
                out_cursor.advance();
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_pages, block_pages);
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("G6C-DRAIN");
        noc_async_write_barrier();
    }
}
