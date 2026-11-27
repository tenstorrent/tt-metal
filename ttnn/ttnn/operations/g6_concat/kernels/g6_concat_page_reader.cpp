// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat page_copy reader (NCRISC / NOC0): every output page is one input page.
// Output page p = o * C + c with Coff_k <= c < Coff_k + C_k reads input k page o * C_k + (c - Coff_k).
// The core walks its output-page range in blocks of block_pages; the (o, k, c') cursor and the
// accessor rebase advance incrementally, so nothing is divided per page.
//
// No kernel_lib helper covers a raw page copy between DRAM and a dataflow CB (op_design.md,
// "Helpers considered and rejected").

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
    constexpr auto input_args = TensorAccessorArgs<5>();

    // Common: block_pages, K, out_Wt, out_Ht, v_last, base[K], C[K].
    const uint32_t block_pages = get_common_arg_val<uint32_t>(0);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(1);
    const uint32_t out_Wt = get_common_arg_val<uint32_t>(2);
    const uint32_t out_Ht = get_common_arg_val<uint32_t>(3);
    const uint32_t v_last = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t base_arg0 = 5;
    const uint32_t pages_arg0 = base_arg0 + num_inputs;

    uint32_t pages_left = get_arg_val<uint32_t>(0);
    uint32_t outer = get_arg_val<uint32_t>(1);
    uint32_t k = get_arg_val<uint32_t>(2);
    uint32_t c_in = get_arg_val<uint32_t>(3);
    uint32_t col = get_arg_val<uint32_t>(4);
    uint32_t tile_row = get_arg_val<uint32_t>(5);

    auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(base_arg0 + k), page_bytes);
    uint32_t pages_k = get_common_arg_val<uint32_t>(pages_arg0 + k);
    uint32_t page_id = outer * pages_k + c_in;

    while (pages_left > 0) {
        const uint32_t extent = pages_left < block_pages ? pages_left : block_pages;
        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-FETCH");
            uint32_t slot = get_write_ptr(cb_output_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                const uint32_t v = g6_concat::valid_rows(tile_row, out_Ht, v_last);
                if (v <= trim_rows_max) {
                    // Rows [0, v) sit at the top of faces 0 and 1, at the same offsets in both pages.
                    const uint64_t src = input.get_noc_addr(page_id);
                    if (v * row_bytes == face_bytes) {
                        noc_async_read(src, slot, 2 * face_bytes);
                    } else {
                        noc_async_read(src, slot, v * row_bytes);
                        noc_async_read(src + face_bytes, slot + face_bytes, v * row_bytes);
                    }
                } else {
                    noc_async_read(input.get_noc_addr(page_id), slot, page_bytes);
                }
                slot += page_bytes;

                ++page_id;
                if (++c_in == pages_k) {
                    c_in = 0;
                    if (++k == num_inputs) {
                        k = 0;
                        ++outer;
                    }
                    input.bank_base_address = get_common_arg_val<uint32_t>(base_arg0 + k);
                    pages_k = get_common_arg_val<uint32_t>(pages_arg0 + k);
                    page_id = outer * pages_k;
                }
                g6_concat::advance_page(col, tile_row, out_Wt, out_Ht);
            }
            noc_async_read_barrier();
        }
        // Always the nominal block: the ring pointers wrap only on an exact fifo_limit hit.
        cb_push_back(cb_output_pages, block_pages);
        pages_left -= extent;
    }
}
