// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat writer (BRISC / NOC1), shared by both regimes: stores the core's contiguous output-page
// range block by block. A page with v <= trim_rows_max valid rows writes only rows [0, v); the
// output's padding rows are unspecified by contract.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_concat_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    // Common: block_pages, out_Wt, out_Ht, v_last, output base.
    const uint32_t block_pages = get_common_arg_val<uint32_t>(0);
    const uint32_t out_Wt = get_common_arg_val<uint32_t>(1);
    const uint32_t out_Ht = get_common_arg_val<uint32_t>(2);
    const uint32_t v_last = get_common_arg_val<uint32_t>(3);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(4), page_bytes);

    uint32_t pages_left = get_arg_val<uint32_t>(0);
    uint32_t page_id = get_arg_val<uint32_t>(1);
    uint32_t col = get_arg_val<uint32_t>(2);
    uint32_t tile_row = get_arg_val<uint32_t>(3);

    while (pages_left > 0) {
        const uint32_t extent = pages_left < block_pages ? pages_left : block_pages;
        {
            DeviceZoneScopedN("G6C-WAIT");
            cb_wait_front(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-STORE");
            uint32_t slot = get_read_ptr(cb_output_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                const uint32_t v = g6_concat::valid_rows(tile_row, out_Ht, v_last);
                const uint64_t dst = output.get_noc_addr(page_id);
                if (v <= trim_rows_max) {
                    if (v * row_bytes == face_bytes) {
                        noc_async_write(slot, dst, 2 * face_bytes);
                    } else {
                        noc_async_write(slot, dst, v * row_bytes);
                        noc_async_write(slot + face_bytes, dst + face_bytes, v * row_bytes);
                    }
                } else {
                    noc_async_write(slot, dst, page_bytes);
                }
                slot += page_bytes;
                ++page_id;
                g6_concat::advance_page(col, tile_row, out_Wt, out_Ht);
            }
            // The slots may be refilled once the data has left L1; completion is awaited once at the end.
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_pages, block_pages);
        pages_left -= extent;
    }
    {
        DeviceZoneScopedN("G6C-DRAIN");
        noc_async_write_barrier();
    }
}
