// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "g6_concat_tile_rows.hpp"

namespace g6_concat {

// Issue order of a run of n consecutive pages (one transfer per page and piece):
//
// * Piece-major. A whole page larger than transfer_bytes moves as transfers of at most
//   transfer_bytes, and the run is swept once per piece: piece 0 of every page, then piece 1, ...
//   Consecutive pages of an interleaved buffer sit in consecutive banks, so consecutive transfers
//   rotate banks instead of streaming page-sized bursts into one bank. On a DRAM-bound mixed
//   read/write stream this is what sets the achieved DRAM rate (TRANSFER_BYTES_MAX on the host).
// * Rotated (bank_stagger). Each sweep starts at in-run index `rotation % n` and wraps, so cores
//   that line up on the same bank start on different ones.
//
// A trimmed page (valid_rows <= trim_rows_max) moves its valid rows once, in the sweep of piece 0.
// Every byte is still moved exactly once; only the order changes.

template <uint32_t page_bytes, uint32_t transfer_bytes>
struct PagePieces {
    static_assert(transfer_bytes > 0 && transfer_bytes <= page_bytes, "a transfer is at most one page");
    static constexpr uint32_t count = (page_bytes + transfer_bytes - 1) / transfer_bytes;
};

// Calls f(j, valid_rows) for the pages j of the run in issue order; returns the cursor one past the
// run. `rows` is the cursor of page 0 of the run.
template <bool rotate, typename F>
FORCE_INLINE OutputRowCursor walk_run(uint32_t n, uint32_t rotation, OutputRowCursor rows, F&& f) {
    if constexpr (!rotate) {
        for (uint32_t j = 0; j < n; ++j) {
            f(j, rows.valid_rows());
            rows.advance();
        }
        return rows;
    } else {
        const OutputRowCursor first = rows;
        for (uint32_t j = 0; j < rotation; ++j) {
            rows.advance();
        }
        for (uint32_t j = rotation; j < n; ++j) {
            f(j, rows.valid_rows());
            rows.advance();
        }
        const OutputRowCursor end = rows;
        rows = first;
        for (uint32_t j = 0; j < rotation; ++j) {
            f(j, rows.valid_rows());
            rows.advance();
        }
        return end;
    }
}

// Moves a run of n pages: whole_piece(j, offset_bytes, size_bytes) per whole page and piece,
// trimmed(j, valid_rows) once per trimmed page. Leaves `rows` one past the run.
template <
    uint32_t page_bytes,
    uint32_t transfer_bytes,
    uint32_t trim_rows_max,
    bool rotate,
    typename WholePiece,
    typename Trimmed>
FORCE_INLINE void sweep_run(
    uint32_t n, uint32_t stagger, OutputRowCursor& rows, WholePiece&& whole_piece, Trimmed&& trimmed) {
    constexpr uint32_t num_pieces = PagePieces<page_bytes, transfer_bytes>::count;
    const uint32_t rotation = rotate ? stagger % n : 0;
    OutputRowCursor end = rows;
    for (uint32_t piece = 0; piece < num_pieces; ++piece) {
        const uint32_t offset = piece * transfer_bytes;
        const uint32_t size = page_bytes - offset < transfer_bytes ? page_bytes - offset : transfer_bytes;
        end = walk_run<rotate>(n, rotation, rows, [&](uint32_t j, uint32_t valid_rows) {
            if (is_trimmed<trim_rows_max>(valid_rows)) {
                if (piece == 0) {
                    trimmed(j, valid_rows);
                }
            } else {
                whole_piece(j, offset, size);
            }
        });
    }
    rows = end;
}

}  // namespace g6_concat
