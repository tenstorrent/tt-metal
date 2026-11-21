// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reshape relayout: regimes tile_row_regroup (RESHAPE_TILE_ROW_REGROUP, op_design.md regroup_rows_block),
// w_regroup_rm (RESHAPE_W_REGROUP_RM, regroup_bytes_block) and w_regroup_tile (RESHAPE_W_REGROUP_TILE,
// regroup_elements_block); the W regimes are described in the last two paragraphs of this comment.
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
//
// w_regroup_rm (row-major, W changes): output row j is bytes [j * page_out, (j + 1) * page_out) of the flat byte
// stream of the input rows (no padding inside it). Per block: wait for the window of the block's input pages (window
// entry e = input page window_first + e, entries entry_in apart), hold it under one scoped_read_lock, and for each
// of the block's output rows reserve one output page and copy the byte runs that fill it -- each run is cut where
// the output page or the current input page ends -- then push it. The input cursor (window page, byte offset in it)
// starts at the host-computed in_off0 of this cluster's first output byte; it is reset to window page 0 at every
// block, because the host makes every block's window start at the page holding the block's first byte. Runs start
// and end at multiples of the element size, at any 16 B phase, so the copy is phase-aware (copy_bytes).
//
// w_regroup_tile (tile layout, W changes): output row g is flat elements [g * w_out, (g + 1) * w_out), output tile
// (k, j) holds columns [32 j, 32 j + 32) of the valid rows of output tile-row k. Per block (output tile-rows of this
// cluster, its output tile columns [col_start, col_start + col_count)): wait for the window -- the input tile-rows
// of the block x the cluster's input tile-column window [win_col0, win_col0 + win_cols), window_cols entries per
// tile-row -- hold it under one scoped_read_lock, and for each output tile-row set one input cursor per valid output
// row (row r + 1 starts w_out elements after row r), then for each output tile column reserve one tile and fill each
// row with the next `cols` elements of its cursor, in element runs cut where an input row, an input face row
// (16 elements) or an output face row ends, then push it. Rows valid..31 and columns past w_out of an output tile
// are not written (unspecified padding, as in tile_row_regroup). The cursor (input tile-row, row in tile, row in
// plane, column) starts at host-computed values and advances by increments, compares and power-of-two shifts.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

namespace {

#if defined(RESHAPE_TILE_ROW_REGROUP) || defined(RESHAPE_W_REGROUP_TILE)
// Tile geometry comes from the host (reshape_program_artifacts.py TILE_ROWS / FACE_ROWS): one source of truth.
constexpr uint32_t TILE_ROWS = get_arg(args::tile_rows);
constexpr uint32_t FACE_ROWS = get_arg(args::face_rows);
#endif

#ifdef RESHAPE_TILE_ROW_REGROUP

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
#endif  // RESHAPE_TILE_ROW_REGROUP

#ifdef RESHAPE_W_REGROUP_TILE
// The input cursor of the element walk: an input element by its tile-row, its row inside that tile, its row inside the
// input plane, and its column (0 .. w_in - 1).
struct Cursor {
    uint32_t tile_row;
    uint32_t row_in_tile;
    uint32_t row_in_plane;
    uint32_t col;
};

template <uint32_t h_in>
inline void next_input_row(Cursor& c) {
    ++c.row_in_plane;
    ++c.row_in_tile;
    if (c.row_in_plane == h_in) {  // a new input plane starts a new input tile-row
        c.row_in_plane = 0;
        c.row_in_tile = 0;
        ++c.tile_row;
    } else if (c.row_in_tile == TILE_ROWS) {
        c.row_in_tile = 0;
        ++c.tile_row;
    }
}

// Move the cursor n elements on in the flat stream (input rows are w_in elements; no divide: one step per row end).
template <uint32_t h_in, uint32_t w_in>
inline void advance(Cursor& c, uint32_t n) {
    for (;;) {
        const uint32_t left = w_in - c.col;
        if (n < left) {
            c.col += n;
            return;
        }
        n -= left;
        c.col = 0;
        next_input_row<h_in>(c);
    }
}

// Byte offset of element (row, col) inside a tile: four FACE_ROWS x FACE_ROWS faces, face rows contiguous.
template <uint32_t face_row_bytes, uint32_t face_bytes, uint32_t elem_bytes>
inline uint32_t element_offset(uint32_t row, uint32_t col) {
    const uint32_t r = row < FACE_ROWS ? row * face_row_bytes : 2 * face_bytes + (row - FACE_ROWS) * face_row_bytes;
    return r + (col < FACE_ROWS ? col * elem_bytes : face_bytes + (col - FACE_ROWS) * elem_bytes);
}
#endif  // RESHAPE_W_REGROUP_TILE

inline void copy_words(volatile uint32_t* dst, const volatile uint32_t* src, uint32_t words) {
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = src[i];
    }
}

// Copy n bytes between two L1 addresses of any phase (the uncached CPU alias: every access is one load or store).
// Same 4-byte phase: byte head, word body, byte tail. Phases 2 apart (the bf16 case of rows cut at odd elements):
// align the destination, then build each destination word from two aligned source words (one load + one store per
// word). The aligned source words read at most 2 bytes outside [src, src + n), inside the same aligned word, so
// inside the 16 B-aligned ring entry. Odd phases (1-byte elements only) fall back to bytes.
inline void copy_bytes(uintptr_t dst, uintptr_t src, uint32_t n) {
    const uintptr_t phase = dst ^ src;
    if ((phase & 3u) == 0) {
        for (; (dst & 3u) != 0 && n != 0; ++dst, ++src, --n) {
            *reinterpret_cast<volatile uint8_t*>(dst) = *reinterpret_cast<const volatile uint8_t*>(src);
        }
        const uint32_t words = n >> 2;
        copy_words(reinterpret_cast<volatile uint32_t*>(dst), reinterpret_cast<const volatile uint32_t*>(src), words);
        dst += words << 2;
        src += words << 2;
        n &= 3u;
    } else if ((phase & 1u) == 0) {
        if ((dst & 1u) != 0 && n != 0) {
            *reinterpret_cast<volatile uint8_t*>(dst++) = *reinterpret_cast<const volatile uint8_t*>(src++);
            --n;
        }
        if ((dst & 2u) != 0 && n >= 2) {
            *reinterpret_cast<volatile uint16_t*>(dst) = *reinterpret_cast<const volatile uint16_t*>(src);
            dst += 2;
            src += 2;
            n -= 2;
        }
        // dst is word aligned and src is 2 mod 4: destination word i is the upper half of source word i (at src - 2)
        // and the lower half of source word i + 1.
        const uint32_t words = n >> 2;
        if (words != 0) {
            const volatile uint32_t* s = reinterpret_cast<const volatile uint32_t*>(src - 2);
            volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst);
            uint32_t lo = s[0];
            for (uint32_t i = 0; i < words; ++i) {
                const uint32_t hi = s[i + 1];
                d[i] = (lo >> 16) | (hi << 16);
                lo = hi;
            }
            dst += words << 2;
            src += words << 2;
            n &= 3u;
        }
    }
    for (; n != 0; ++dst, ++src, --n) {
        *reinterpret_cast<volatile uint8_t*>(dst) = *reinterpret_cast<const volatile uint8_t*>(src);
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
#elif defined(RESHAPE_W_REGROUP_RM)
    const uint32_t row_count = get_arg(args::row_count);
    uint32_t in_off = get_arg(args::in_off0);  // byte offset of the cursor in its input page
    constexpr uint32_t block_out_rows = get_arg(args::block_out_rows);
    constexpr uint32_t window_entries = get_arg(args::window_entries);
    constexpr uint32_t page_in = get_arg(args::page_in);    // logical bytes of an input row
    constexpr uint32_t page_out = get_arg(args::page_out);  // logical bytes of an output row
    constexpr uint32_t entry_in = get_arg(args::entry_in);  // window entry stride
    static_assert(get_arg(args::relayout_threads) == 1, "the window is consumed by one relayout thread");

    DataflowBuffer window(dfb::window);
    DataflowBuffer tiles(dfb::tiles);

    for (uint32_t kb = 0; kb < row_count; kb += block_out_rows) {
        const uint32_t nr = kb + block_out_rows < row_count ? block_out_rows : row_count - kb;
        // regroup_bytes_block(block_idx)
        window.wait_front(window_entries);
        {
            auto window_lock = window.scoped_read_lock(window_entries);
            // The cursor's input page: the window's first page at every block start.
            uintptr_t src_page = reinterpret_cast<uintptr_t>(&window_lock.get_ptr<uint32_t>()[0]);
            for (uint32_t r = 0; r < nr; ++r) {
                tiles.reserve_back(1);
                {
                    auto page_lock = tiles.scoped_write_lock(1);
                    const uintptr_t dst = reinterpret_cast<uintptr_t>(&page_lock.get_ptr<uint32_t>()[0]);
                    for (uint32_t dst_off = 0; dst_off < page_out;) {
                        const uint32_t src_left = page_in - in_off;
                        const uint32_t dst_left = page_out - dst_off;
                        const uint32_t len = dst_left < src_left ? dst_left : src_left;
                        copy_bytes(dst + dst_off, src_page + in_off, len);
                        dst_off += len;
                        in_off += len;
                        if (in_off == page_in) {  // the input page is used up: the next one is the next entry
                            in_off = 0;
                            src_page += entry_in;
                        }
                    }
                }
                tiles.push_back(1);
            }
        }
        window.pop_front(window_entries);
    }
#elif defined(RESHAPE_W_REGROUP_TILE)
    const uint32_t col_count = get_arg(args::col_count);          // output tile columns of this cluster
    const uint32_t row_count = get_arg(args::row_count);          // output tile-rows of this cluster
    const uint32_t win_col0 = get_arg(args::win_col0);            // input tile column of window column 0
    const uint32_t last_tile_cols = get_arg(args::last_tile_cols);  // valid columns of the last output tile column
    Cursor cur{get_arg(args::in_tile_row0), get_arg(args::row_in_tile0), get_arg(args::row_in_plane0), get_arg(args::col_in0)};
    uint32_t out_row = get_arg(args::out_row0);  // first row of the current output tile-row, inside its plane
    constexpr uint32_t block_out_tile_rows = get_arg(args::block_out_tile_rows);
    constexpr uint32_t window_entries = get_arg(args::window_entries);
    constexpr uint32_t window_cols = get_arg(args::window_cols);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t h_in = get_arg(args::h_in);
    constexpr uint32_t w_in = get_arg(args::w_in);
    constexpr uint32_t h_out = get_arg(args::h_out);
    constexpr uint32_t w_out = get_arg(args::w_out);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);
    constexpr uint32_t face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    static_assert(get_arg(args::relayout_threads) == 1, "the window is consumed by one relayout thread");
    static_assert((TILE_ROWS & (TILE_ROWS - 1)) == 0, "column -> tile column is a shift");

    DataflowBuffer window(dfb::window);
    DataflowBuffer tiles(dfb::tiles);
    Cursor rows[TILE_ROWS];  // per row of the current output tile-row: where its next element is in the input

    for (uint32_t kb = 0; kb < row_count; kb += block_out_tile_rows) {
        const uint32_t nr = kb + block_out_tile_rows < row_count ? block_out_tile_rows : row_count - kb;
        // regroup_elements_block(block_idx)
        window.wait_front(window_entries);
        {
            auto window_lock = window.scoped_read_lock(window_entries);
            // Window entry (t, w) holds input tile (window_first + t, win_col0 + w); fold win_col0 into the origin.
            const uintptr_t window_origin =
                reinterpret_cast<uintptr_t>(&window_lock.get_ptr<uint32_t>()[0]) - win_col0 * page_bytes;
            const uint32_t window_first = cur.tile_row;  // the block's first element lies in the window's first tile-row
            for (uint32_t k = 0; k < nr; ++k) {
                const uint32_t valid = h_out - out_row < TILE_ROWS ? h_out - out_row : TILE_ROWS;
                // Output row r + 1 starts w_out elements after output row r (padding rows are not elements).
                rows[0] = cur;
                for (uint32_t r = 1; r < valid; ++r) {
                    rows[r] = rows[r - 1];
                    advance<h_in, w_in>(rows[r], w_out);
                }
                cur = rows[valid - 1];
                advance<h_in, w_in>(cur, w_out);  // the next output tile-row's first row
                out_row += valid;
                if (out_row == h_out) {
                    out_row = 0;
                }
                // Each output tile takes the next `cols` elements of each of its rows: element runs cut where an input
                // row, an input face row or an output face row ends.
                for (uint32_t j = 0; j < col_count; ++j) {
                    const uint32_t cols = j + 1 < col_count ? TILE_ROWS : last_tile_cols;
                    tiles.reserve_back(1);
                    {
                        auto tile_lock = tiles.scoped_write_lock(1);
                        const uintptr_t dst_tile = reinterpret_cast<uintptr_t>(&tile_lock.get_ptr<uint32_t>()[0]);
                        for (uint32_t r = 0; r < valid; ++r) {
                            Cursor& c = rows[r];
                            for (uint32_t oc = 0; oc < cols;) {
                                const uint32_t in_col = c.col % TILE_ROWS;
                                uint32_t len = oc < FACE_ROWS ? FACE_ROWS - oc : cols - oc;
                                len = cols - oc < len ? cols - oc : len;
                                const uint32_t in_face_left = in_col < FACE_ROWS ? FACE_ROWS - in_col : TILE_ROWS - in_col;
                                len = in_face_left < len ? in_face_left : len;
                                len = w_in - c.col < len ? w_in - c.col : len;
                                const uintptr_t src =
                                    window_origin +
                                    ((c.tile_row - window_first) * window_cols + c.col / TILE_ROWS) * page_bytes +
                                    element_offset<face_row_bytes, face_bytes, elem_bytes>(c.row_in_tile, in_col);
                                const uintptr_t dst =
                                    dst_tile + element_offset<face_row_bytes, face_bytes, elem_bytes>(r, oc);
                                copy_bytes(dst, src, len * elem_bytes);
                                oc += len;
                                c.col += len;
                                if (c.col == w_in) {
                                    c.col = 0;
                                    next_input_row<h_in>(c);
                                }
                            }
                        }
                    }
                    tiles.push_back(1);
                }
            }
        }
        window.pop_front(window_entries);
    }
#endif
}
