// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat page_copy reader (NCRISC / NOC0).
// page_fetch_block: block_pages consecutive output pages per block, each one input page. Output
// page p = o*C + c with Coff_k <= c < Coff_k + C_k comes from input k page o*C_k + (c - Coff_k).
// The walk cursor (o, k, c') starts from per-core runtime args and only increments, so no
// division happens per page. Pages whose tile row holds v <= TRIM_ROWS_MAX valid rows move only
// rows [0, v), at identical in-tile offsets on both sides (row shift 0, always alignment-legal).
//
// No kernel_lib helper covers an interleaved page fetch into a CB (see op_design.md, "Helpers
// considered and rejected"), so this kernel uses raw dataflow calls.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_concat_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(2);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_l1_mask_lo = get_compile_time_arg_val(5);
    constexpr uint32_t input_l1_mask_hi = get_compile_time_arg_val(6);
    constexpr auto dram_input_args = TensorAccessorArgs<7>();
    constexpr auto l1_input_args = TensorAccessorArgs<dram_input_args.next_compile_time_args_offset()>();
    static_assert(trim_rows_max <= g6_concat::FACE_ROWS, "row trimming only covers faces 0 and 1");

    // Common runtime args: block_pages, K, out_Wt, out_Ht, v_last, then (base_addr_k, C_k) per input.
    constexpr uint32_t INPUT_ARGS_BASE = 5;
    const uint32_t block_pages = get_common_arg_val<uint32_t>(0);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(1);
    g6_concat::OutputPageCursor out_cursor{
        .out_Wt = get_common_arg_val<uint32_t>(2),
        .out_Ht = get_common_arg_val<uint32_t>(3),
        .v_last = get_common_arg_val<uint32_t>(4),
        .R = 0,
        .c = 0};
    auto input_addr = [&](uint32_t k) { return get_common_arg_val<uint32_t>(INPUT_ARGS_BASE + 2 * k); };
    auto input_pages_per_slice = [&](uint32_t k) { return get_common_arg_val<uint32_t>(INPUT_ARGS_BASE + 2 * k + 1); };

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t o = get_arg_val<uint32_t>(2);
    uint32_t k = get_arg_val<uint32_t>(3);
    uint32_t c_in = get_arg_val<uint32_t>(4);

    out_cursor.init(start_page);
    g6_concat::InputPages<decltype(dram_input_args), decltype(l1_input_args), input_l1_mask_lo, input_l1_mask_hi>
        inputs(dram_input_args, l1_input_args, page_bytes);
    uint32_t C_k = input_pages_per_slice(k);
    inputs.rebase(k, input_addr(k));
    uint32_t in_page = o * C_k + c_in;

    const uint32_t num_blocks = (num_pages + block_pages - 1) / block_pages;
    uint32_t remaining = num_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Ragged last block: nominal reserve/push of block_pages, only `extent` pages transferred.
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-FETCH");
            uint32_t slot = get_write_ptr(cb_output_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                const uint32_t v = out_cursor.valid_rows();
                if (v <= trim_rows_max) {
                    g6_concat::for_each_leading_rows_range(
                        v, face_row_bytes, face_bytes, [&](uint32_t offset, uint32_t bytes) {
                            noc_async_read(inputs.noc_addr(in_page, offset), slot + offset, bytes);
                        });
                } else {
                    noc_async_read(inputs.noc_addr(in_page, 0), slot, page_bytes);
                }
                slot += page_bytes;
                out_cursor.advance();

                ++in_page;
                if (++c_in == C_k) {
                    c_in = 0;
                    if (++k == num_inputs) {
                        k = 0;
                        ++o;
                    }
                    C_k = input_pages_per_slice(k);
                    inputs.rebase(k, input_addr(k));
                    in_page = o * C_k;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_output_pages, block_pages);
        remaining -= extent;
    }
}
