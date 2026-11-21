// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reshape relayout: regime tile_row_regroup only (op_design.md, regroup_rows_block).
//
// This is a DATA-MOVEMENT kernel on a DM hart, not a NEO compute kernel: reshape computes nothing (spec rule R1),
// and the one transforming stage of the op -- moving rows between tiles when H changes in tile layout -- is a
// byte copy. It is named reshape_compute.cpp because it is the middle (transforming) stage of the pipeline.
//
// Helper substitution (raw CPU copy instead of a kernel-library helper), from op_design.md "Helpers considered":
//   - dataflow_kernel_lib::local_addr + self-aimed Noc::async_read (local_copy_helpers_dataflow.hpp): the copy's
//     source is a byte offset inside the input window, and the window's NoC address is private
//     (dataflow_buffer.h get_noc_read_addr); its public read pointer is the uncached CPU alias.
//   - set_read_state / read_with_state: single packet with a compile-time size; run lengths vary per run.
//   - tile_column_dataflow.hpp: moves columns, not contiguous face-row spans.
//   - compute_kernel_lib::untilize / tilize: cannot drop or insert padding rows; a DM row move would remain.
//
// Per block (column block outer, row block inner): wait for the whole input window (nominal window_entries, so it
// never wraps), hold it under one scoped_read_lock, and for each output tile-row k of the block build one run table
// -- maximal row spans that stay inside one input tile and one face row-half on both sides -- then for each of the
// block's tile columns reserve one output tile, copy every run (left faces 0/2 and right faces 1/3: two contiguous
// spans of len * face_row_bytes each), push it. Rows valid_rows(k)..31 of an output tile are not written (their
// content is unspecified: spec DECISIONS, pad_value=None).
//
// The input cursor (input tile-row, row in that tile, row in the input plane) and the output row in its plane start
// at host-computed values and advance by increments and compares only: the kernel never divides.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

struct Run {
    uint32_t src_tile_row;  // input tile-row, relative to the window's first
    uint32_t src_row;       // first source row inside that tile
    uint32_t dst_row;       // first destination row inside the output tile
    uint32_t len;           // rows
};

template <uint32_t face_row_bytes, uint32_t face_bytes>
inline uint32_t row_offset(uint32_t row) {
    // Row r of a tile: its left half is in face 0 (r < 16) or face 2, its right half one face further.
    return row < FACE_ROWS ? row * face_row_bytes : 2 * face_bytes + (row - FACE_ROWS) * face_row_bytes;
}

inline void copy_words(volatile uint32_t* dst, const volatile uint32_t* src, uint32_t words) {
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = src[i];
    }
}

}  // namespace

void kernel_main() {
#ifdef RESHAPE_TILE_ROW_REGROUP
    const uint32_t col_count = get_arg(args::col_count);
    const uint32_t row_count = get_arg(args::row_count);
    const uint32_t in_tile_row0 = get_arg(args::in_tile_row0);
    const uint32_t row_in_tile0 = get_arg(args::row_in_tile0);
    const uint32_t row_in_plane0 = get_arg(args::row_in_plane0);
    const uint32_t out_row0 = get_arg(args::out_row0);
    constexpr uint32_t block_tile_cols = get_arg(args::block_tile_cols);
    constexpr uint32_t block_out_tile_rows = get_arg(args::block_out_tile_rows);
    constexpr uint32_t window_entries = get_arg(args::window_entries);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t h_in = get_arg(args::h_in);
    constexpr uint32_t h_out = get_arg(args::h_out);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);
    constexpr uint32_t face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);  // compile-time constant: no libgcc
    constexpr uint32_t tile_row_stride_bytes = block_tile_cols * page_bytes;  // window: one input tile-row
    static_assert(get_arg(args::relayout_threads) == 1, "the window is consumed by one relayout thread");
    static_assert(face_row_bytes % sizeof(uint32_t) == 0, "face rows are copied as 32-bit words");

    DataflowBuffer window(dfb::window);
    DataflowBuffer tiles(dfb::tiles);
    Run runs[TILE_ROWS];

    for (uint32_t cb = 0; cb < col_count; cb += block_tile_cols) {
        const uint32_t wb = cb + block_tile_cols < col_count ? block_tile_cols : col_count - cb;
        // Every column block walks the same rows: reset the cursors to this cluster's first row.
        uint32_t in_tile_row = in_tile_row0;
        uint32_t row_in_tile = row_in_tile0;
        uint32_t row_in_plane = row_in_plane0;
        uint32_t out_row = out_row0;
        for (uint32_t kb = 0; kb < row_count; kb += block_out_tile_rows) {
            const uint32_t nr = kb + block_out_tile_rows < row_count ? block_out_tile_rows : row_count - kb;
            // regroup_rows_block(block_idx)
            window.wait_front(window_entries);
            {
                auto window_lock = window.scoped_read_lock(window_entries);
                const uintptr_t window_base = reinterpret_cast<uintptr_t>(&window_lock.get_ptr<uint32_t>()[0]);
                const uint32_t window_first = in_tile_row;  // the block's first row lies in the window's first tile-row
                for (uint32_t k = 0; k < nr; ++k) {
                    const uint32_t valid = h_out - out_row < TILE_ROWS ? h_out - out_row : TILE_ROWS;
                    // The run table of this output tile-row.
                    uint32_t num_runs = 0;
                    for (uint32_t dst = 0; dst < valid;) {
                        uint32_t len = valid - dst;
                        const uint32_t src_left = h_in - row_in_plane;  // rows left in the input plane ...
                        const uint32_t tile_left = TILE_ROWS - row_in_tile;  // ... and in the input tile
                        const uint32_t src_half = row_in_tile < FACE_ROWS ? FACE_ROWS - row_in_tile : tile_left;
                        const uint32_t dst_half = dst < FACE_ROWS ? FACE_ROWS - dst : TILE_ROWS - dst;
                        len = len < src_left ? len : src_left;
                        len = len < src_half ? len : src_half;
                        len = len < dst_half ? len : dst_half;
                        runs[num_runs++] = {in_tile_row - window_first, row_in_tile, dst, len};
                        dst += len;
                        row_in_tile += len;
                        row_in_plane += len;
                        if (row_in_plane == h_in) {  // a new input plane starts a new input tile-row
                            row_in_plane = 0;
                            row_in_tile = 0;
                            ++in_tile_row;
                        } else if (row_in_tile == TILE_ROWS) {
                            row_in_tile = 0;
                            ++in_tile_row;
                        }
                    }
                    out_row += valid;
                    if (out_row == h_out) {
                        out_row = 0;
                    }
                    // Build this tile-row's wb output tiles from the run table.
                    for (uint32_t w = 0; w < wb; ++w) {
                        tiles.reserve_back(1);
                        {
                            auto tile_lock = tiles.scoped_write_lock(1);
                            const uintptr_t dst_tile = reinterpret_cast<uintptr_t>(&tile_lock.get_ptr<uint32_t>()[0]);
                            const uintptr_t src_col = window_base + w * page_bytes;
                            for (uint32_t r = 0; r < num_runs; ++r) {
                                const Run& run = runs[r];
                                const uintptr_t src = src_col + run.src_tile_row * tile_row_stride_bytes +
                                                     row_offset<face_row_bytes, face_bytes>(run.src_row);
                                const uintptr_t dst = dst_tile + row_offset<face_row_bytes, face_bytes>(run.dst_row);
                                const uint32_t words = run.len * face_row_words;
                                copy_words(reinterpret_cast<volatile uint32_t*>(dst),
                                           reinterpret_cast<const volatile uint32_t*>(src), words);
                                copy_words(reinterpret_cast<volatile uint32_t*>(dst + face_bytes),
                                           reinterpret_cast<const volatile uint32_t*>(src + face_bytes), words);
                            }
                        }
                        tiles.push_back(1);
                    }
                }
            }
            window.pop_front(window_entries);
        }
    }
#endif
}
