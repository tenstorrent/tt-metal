// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tiled geometry: the layout contract between the program factory, the V2
// reader and the compute kernel when compute builds the sampling geometry of a
// whole output block from tiles, instead of the reader handing it over one
// point at a time.
//
// Point j = l * P + p of a block lives in SFPU lane j: column j of the left
// face and column 16 + j of the right one. px_j and ref_x go left, py_j and
// ref_y right; the block solve in msda_geometry.hpp places the rest by its own
// table. That is why the block must have exactly POINTS points.
//
//   offsets tile   row q = the packed offsets of query q: column 2j = x_j,
//                  2j + 1 = y_j
//   attn tile      row q = attention_weights of query q in columns 0..15;
//                  columns 16..31 hold whatever the aligned attn read brought
//                  in, which the block solve zeroes before packing
//   ref tile       (q, k) = ref_x[k], (q, 16 + k) = ref_y[k], (q, BIAS_COL) = 1
//   M_off, M_ref   px | py = offsets @ M_off + ref @ M_ref, with the per-level
//                  scales and bias as entries. The FPU applies them exactly
//                  only because each is bf16-exact (the factory checks) and the
//                  kernel runs at HiFi4: lower fidelities drop SrcB mantissa
//                  bits, and in0 (the data) is SrcB
//   E_j            1 at (j, 0): w @ E_j moves column j of a weight tile w to
//                  column 0, where the COL broadcast of the reduction reads
//                  its scalar
//
// Rows q >= v_rows of the staged page and the ref tile are never rewritten:
// they hold a previous block's values (zero at first), so their weights are not
// zero. That is harmless: every row is solved, reduced and accumulated on its
// own, and the writer emits only rows < v_rows.
//
// Shared with the program factory: plain constants only.

#pragma once

#include <cstdint>

namespace fused_msda_tiled_geometry {

constexpr uint32_t POINTS = 16;  // L * P; one face width
// Reference tile columns 0..MAX_REFS-1 carry x, 16.. carry y; BIAS_COL stays 1.
constexpr uint32_t MAX_REFS = 15;
constexpr uint32_t BIAS_COL = 31;
static_assert(16 + MAX_REFS <= BIAS_COL, "ref y columns must not reach the bias column");

// Tile order of the constants CB, written once by the reader and never popped.
constexpr uint32_t CONST_M_OFF = 0;
constexpr uint32_t CONST_M_REF = 1;
constexpr uint32_t CONST_E_FIRST = 2;
constexpr uint32_t CONST_TILES = CONST_E_FIRST + POINTS;

// One block's staged operands, row-major, tilized by compute into OFF_TILE and
// ATTN_TILE: row q = [offsets (64 B) | attention weights (32 B) | pad (32 B)].
constexpr uint32_t BF16_NBYTES = 2;
constexpr uint32_t TILE_ROW_NBYTES = 32 * BF16_NBYTES;
constexpr uint32_t RM_TILES = 2;
constexpr uint32_t RM_ROW_NBYTES = RM_TILES * TILE_ROW_NBYTES;
constexpr uint32_t RM_ATTN_OFFSET = POINTS * 2 * BF16_NBYTES;  // the offsets stick
constexpr uint32_t OFF_TILE = 0;
constexpr uint32_t ATTN_TILE = 1;
static_assert(RM_ATTN_OFFSET == TILE_ROW_NBYTES, "the offsets stick must fill exactly one tile row");

// CB depths, in pages. The reader stages block t+1 only after block t's x0/y0
// arrive, and compute pops block t's staged page before pushing them, so one
// block of staging is enough. The reader holds block t's x0/y0 for the whole
// block while compute solves block t+1, and compute solves block t+1 while
// still extracting block t's weights, so those hold two blocks. The reference
// tile is one page so that a block whose reference points did not change can
// push it again without rewriting it.
constexpr uint32_t RM_CB_PAGES = RM_TILES;
constexpr uint32_t CORNER_CB_PAGES = 2;
constexpr uint32_t WEIGHT_TILES = 4;  // w_NW, w_NE, w_SW, w_SE
constexpr uint32_t WEIGHT_CB_PAGES = 2 * WEIGHT_TILES;
constexpr uint32_t REF_TILE_CB_PAGES = 1;

// Per-level constants the reader writes into M_off / M_ref, as fp32 bit
// patterns in its runtime args: (primary, secondary, bias) for x then y.
constexpr uint32_t CONSTANTS_PER_LEVEL = 6;

}  // namespace fused_msda_tiled_geometry
