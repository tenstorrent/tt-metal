// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC, NoC0).
//
// Per segment (one token-tile row of this core's unit range):
//   load_block        — per block of block_col_tiles columns: B F tiles (slot c) and n*B X tiles
//                       (slot i*B + c), valid columns only, ONE barrier, nominal pushes.
//   load_coefficients — only when the host's COEF_EXPANDER knob names the reader (expand_here); otherwise the
//                       writer owns it (mhc_post_coef_expand.hpp). Here the raw post / comb reads ride block 0's
//                       barrier, and the expansion runs in the read shadow of block 1 (block 0 is already
//                       pushed; compute starts on it as soon as stream 0's coefficients land). A single-block
//                       segment expands right after block 0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"
#include "mhc_skip_noc.hpp"

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t n = get_compile_time_arg_val(0);                   // streams
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);   // Ct = C / 32
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);     // B
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(3);  // ceil(n / 32)
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(4);  // ceil(n*n / 32)
    constexpr uint32_t sublayer_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t residual_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t coef_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(8);  // 32
    constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(11);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(12);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(13);  // P = ceil((n+1)/2)
    constexpr bool expand_here = get_compile_time_arg_val(14) != 0;           // COEF_EXPANDER == reader
    constexpr auto sublayer_args = TensorAccessorArgs<15>();
    constexpr auto residual_args = TensorAccessorArgs<sublayer_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr uint32_t residual_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t residual_block_tiles = n * block_col_tiles;

    // ---- runtime args ----
    const uint32_t sublayer_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    const auto sublayer_acc = TensorAccessor(sublayer_args, sublayer_addr, sublayer_page_bytes);
    const auto residual_acc = TensorAccessor(residual_args, residual_addr, residual_page_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page_bytes);
    mhc_post::CoefExpander<
        n,
        coef_tiles_per_stream,
        post_tiles_per_row,
        comb_tiles_per_row,
        coef_page_bytes,
        cb_coef_raw,
        cb_coef_bcast,
        decltype(post_acc)>
        coefs(post_acc, comb_acc);

    // Issue the async reads of one data block (no barrier) into already-reserved CB windows.
    auto issue_block_reads = [&](uint32_t row, uint32_t col_start, uint32_t valid_cols) {
        const uint32_t f_base = get_write_ptr(cb_sublayer_tiles);
        const uint32_t x_base = get_write_ptr(cb_residual_tiles);
        const uint32_t f_page0 = row * col_tiles_per_row + col_start;
        for (uint32_t c = 0; c < valid_cols; ++c) {
            data_read(sublayer_acc.get_noc_addr(f_page0 + c), f_base + c * sublayer_page_bytes, sublayer_page_bytes);
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x_page0 = row * residual_row_tiles + i * col_tiles_per_row + col_start;
            const uint32_t x_slot0 = x_base + i * block_col_tiles * residual_page_bytes;
            for (uint32_t c = 0; c < valid_cols; ++c) {
                data_read(
                    residual_acc.get_noc_addr(x_page0 + c), x_slot0 + c * residual_page_bytes, residual_page_bytes);
            }
        }
    };

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        const uint32_t blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);
        const uint32_t expand_block = blocks > 1 ? 1 : 0;

        for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
            const uint32_t valid = mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, block_idx);
            const uint32_t col_start = seg.col0 + block_idx * block_col_tiles;

            {
                MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
                cb_reserve_back(cb_sublayer_tiles, block_col_tiles);
                cb_reserve_back(cb_residual_tiles, residual_block_tiles);
            }
            {
                MaybeDeviceZoneScope("reader_issue");
                if constexpr (expand_here) {
                    if (block_idx == 0) {
                        coefs.issue_raw_reads(seg.row);  // shares block 0's barrier
                    }
                }
                issue_block_reads(seg.row, col_start, valid);
            }
            if constexpr (expand_here) {
                if (block_idx == expand_block && block_idx > 0) {
                    MaybeDeviceZoneScope("reader_coef_expand");
                    coefs.expand();  // raw tiles landed with block 0; in block 1's read shadow
                }
            }
            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_sublayer_tiles, block_col_tiles);
            cb_push_back(cb_residual_tiles, residual_block_tiles);
            if constexpr (expand_here) {
                if (block_idx == expand_block && block_idx == 0) {
                    MaybeDeviceZoneScope("reader_coef_expand");
                    coefs.expand();  // single-block segment
                }
            }
        }
    }
}
