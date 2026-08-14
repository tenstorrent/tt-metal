// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize pad-fill primitives, shared by the reader and the writer.
//
// The pad fill is materialized TWICE, in two different element formats, and the
// two are not redundant (op_design.md §10, Refinement 4):
//
//   * the READER fills the row-major input CB in the **input** element format,
//     as it assembles each block. That is the hard contract — the fill travels
//     the same path as real data, so packing it in `output_dtype` would be
//     garbage the moment a cast is requested. It is also the only fill needed
//     whenever the format round-trip is exact (no cast, or a cast that cannot
//     perturb the value).
//
//   * the WRITER re-stamps the pad region of each finished output tile in the
//     **output** element format, AFTER the cast. This is what makes a WIDENING
//     cast exact: a fill that is inexact in the input format (10.2 in bfloat16)
//     would otherwise arrive input-rounded in a wider output (10.1875 in an
//     fp32 tensor) while the oracle expects the output-format value. The host
//     enables it only when the round-trip actually loses the value, so the hot
//     path compiles to nothing.

#pragma once

#include <cstdint>
#include <type_traits>

namespace tilize_kernels {

// Alignment-aware L1 fill: 4-byte stores for the aligned middle, element-sized
// stores for the unaligned head/tail (rv32 faults on unaligned wide stores).
// `val` carries the fill in the target element format in its low bytes; it is
// replicated across the 32-bit store word, so a sub-word element fills every
// position (a value written once per word is invisible at 0 and garbage
// otherwise).
template <uint32_t elem_bytes>
FORCE_INLINE void fill_l1_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4, "unsupported element width");
    using elem_t =
        std::conditional_t<elem_bytes == 1, uint8_t, std::conditional_t<elem_bytes == 2, uint16_t, uint32_t>>;

    const uint32_t end_addr = start_addr + n_bytes;
    const uint32_t start_addr_4B = (start_addr + 3u) & ~3u;
    const uint32_t end_addr_4B = end_addr & ~3u;

    uint32_t val_4B = val;
    if constexpr (elem_bytes == 1) {
        const uint32_t b = val & 0xFFu;
        val_4B = (b << 24) | (b << 16) | (b << 8) | b;
    } else if constexpr (elem_bytes == 2) {
        const uint32_t h = val & 0xFFFFu;
        val_4B = (h << 16) | h;
    }

    for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
         ptr < reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
         ++ptr) {
        *ptr = val_4B;
    }

    if constexpr (elem_bytes < 4) {
        const elem_t v = static_cast<elem_t>(val);
        for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr);
             ptr < reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr_4B);
             ++ptr) {
            *ptr = v;
        }
        for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr_4B);
             ptr < reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr);
             ++ptr) {
            *ptr = v;
        }
    }
}

// Re-stamp the pad region of ONE **tiled** output tile at `tile_addr` with `word`
// (packed in the OUTPUT element format).
//
// `valid_rows` / `valid_cols` are the leading extents this tile carries real data
// over; every position outside `[0, valid_rows) x [0, valid_cols)` is pad. A whole
// pad tile is `valid_rows == 0` (or `valid_cols == 0`), and an untouched tile is
// `valid_rows == tile_h && valid_cols == tile_w`, which costs nothing.
//
// Tiled layout: FACE_H x FACE_W faces stored one after another (face-row-major
// over the tile), row-major inside a face. The loop is written FACE-FIRST rather
// than row-first so it fills the LARGEST contiguous runs the geometry allows —
// which is what makes the stamp affordable:
//
//   * a whole pad TILE   -> ONE run of tile_h*tile_w elements
//   * a whole pad FACE   -> one run of 256 elements
//   * the rows below the data inside a partly-valid face -> one run
//   * only a W tail on a valid row costs a per-row run
//
// Measured (test_bench_lever_out_fill, worst-case geometry: half the output tiles
// are whole pad tiles): the face-first form is FLAT against a row-at-a-time one.
// The cost is the raw STORE COUNT, not the loop overhead — an rv32 volatile L1
// word store is ~10 cycles, and the pad region here is 2 M fp32 elements. Fewer,
// bigger runs is still the right shape (strictly fewer instructions, and it is
// what lets the reader's now-redundant fill be skipped), but do not expect the
// run length itself to buy anything.
template <uint32_t tile_h, uint32_t tile_w, uint32_t elem_bytes>
FORCE_INLINE void fill_tile_pad(uint32_t tile_addr, uint32_t valid_rows, uint32_t valid_cols, uint32_t word) {
    constexpr uint32_t FACE_H = 16;
    constexpr uint32_t FACE_W = 16;
    static_assert(tile_h % FACE_H == 0 && tile_w % FACE_W == 0, "fill_tile_pad needs whole 16x16 faces");
    constexpr uint32_t FACES_PER_ROW = tile_w / FACE_W;
    constexpr uint32_t FACE_ROWS = tile_h / FACE_H;
    constexpr uint32_t FACE_ELEMS = FACE_H * FACE_W;

    if (valid_rows == 0 || valid_cols == 0) {
        // A WHOLE pad tile — the dominant case whenever the pad target exceeds the
        // tile-rounded shape. Faces are contiguous, so this is a single run.
        fill_l1_with_val<elem_bytes>(tile_addr, tile_h * tile_w * elem_bytes, word);
        return;
    }
    if (valid_rows >= tile_h && valid_cols >= tile_w) {
        return;  // fully inside the data region — nothing to stamp
    }

    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t vr = (valid_rows > fr * FACE_H) ? (valid_rows - fr * FACE_H) : 0;  // valid rows in this face
        if (vr > FACE_H) {
            vr = FACE_H;
        }
        for (uint32_t fc = 0; fc < FACES_PER_ROW; ++fc) {
            uint32_t vc = (valid_cols > fc * FACE_W) ? (valid_cols - fc * FACE_W) : 0;  // valid cols in this face
            if (vc > FACE_W) {
                vc = FACE_W;
            }
            const uint32_t face_addr = tile_addr + (fr * FACES_PER_ROW + fc) * FACE_ELEMS * elem_bytes;
            if (vr == 0 || vc == 0) {
                fill_l1_with_val<elem_bytes>(face_addr, FACE_ELEMS * elem_bytes, word);  // whole face is pad
                continue;
            }
            if (vc < FACE_W) {  // W tail of each valid row (the only per-row cost)
                for (uint32_t rr = 0; rr < vr; ++rr) {
                    fill_l1_with_val<elem_bytes>(
                        face_addr + (rr * FACE_W + vc) * elem_bytes, (FACE_W - vc) * elem_bytes, word);
                }
            }
            if (vr < FACE_H) {  // every row below the data, in one contiguous run
                fill_l1_with_val<elem_bytes>(
                    face_addr + vr * FACE_W * elem_bytes, (FACE_H - vr) * FACE_W * elem_bytes, word);
            }
        }
    }
}

}  // namespace tilize_kernels
