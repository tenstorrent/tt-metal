// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tiled geometry: the layout contract between the program factory, the V2
// reader and the compute kernel when compute builds the sampling geometry of a
// whole output block from tiles, instead of the reader handing it over one
// point at a time.
//
// A block's points J = l * P + p go in groups of LANES: point J is lane
// j = J % LANES of group g = J / LANES. Lane j is column j of the left face and
// column 16 + j of the right one, the same SFPU lane, so one DST window solves
// one group. px_j and ref_x go left, py_j and ref_y right; the block solve in
// msda_geometry.hpp places the rest by its own table. The last group's lanes
// past the block's last point are unused: their constants are zero and no
// point is ever extracted from them.
//
//   offsets tile g row q = the packed offsets of points 16g .. 16g + 15 of
//                  query q: column 2j = x_j, 2j + 1 = y_j
//   attn tile      row q = attention_weights of query q, point J in column J;
//                  group 1 reads its weights from the right face. Columns past
//                  the last point hold whatever the aligned read brought in
//   ref tile       (q, k) = ref_x[k], (q, 16 + k) = ref_y[k], (q, BIAS_COL) = 1;
//                  float32 when the reference points are, split by compute into
//                  FP32_REF_PARTS bf16 tiles that sum to it exactly
//   M_off_g,       px | py of group g = offsets_g @ M_off_g + ref @ M_ref_g,
//   M_ref_g        with the per-level scales and bias as entries. The FPU
//                  applies them exactly only because each is bf16-exact (the
//                  factory checks) and the kernel runs at HiFi4: lower
//                  fidelities drop low mantissa bits of SrcB (the data) or SrcA
//                  (the constants)
//   M_extent_g     row BIAS_COL = H_l(j) in column j, W_l(j) in 16 + j, so
//                  ref @ M_extent_g = H_j | W_j on every row (the corner word's
//                  bounds, msda_corner_word.hpp)
//   E_j            1 at (j, 0): w @ E_j moves column j of a weight tile w to
//                  column 0, where the COL broadcast of the reduction reads
//                  its scalar
//
// An operand column no point uses meets a zero row of its matrix. Blackhole's
// FPU multiplies inf and NaN by zero to zero, so whatever such a column holds
// (alignment padding included) adds nothing.
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

constexpr uint32_t LANES = 16;  // points per group; one face width
// The attention weights of a block are one tile row of bf16.
constexpr uint32_t MAX_GROUPS = 2;
constexpr uint32_t MAX_POINTS = MAX_GROUPS * LANES;
// Reference tile columns 0..MAX_REFS-1 carry x, 16.. carry y; BIAS_COL stays 1.
constexpr uint32_t MAX_REFS = 15;
constexpr uint32_t BIAS_COL = 31;
static_assert(16 + MAX_REFS <= BIAS_COL, "ref y columns must not reach the bias column");

constexpr uint32_t groups(uint32_t points) { return (points + LANES - 1) / LANES; }
constexpr uint32_t group_of(uint32_t point) { return point / LANES; }
constexpr uint32_t lane_of(uint32_t point) { return point % LANES; }
// The compute kernel solves block t + 1 while extracting point
// SOLVE_NEXT_BLOCK_AT of block t, so a block needs a point past it.
constexpr uint32_t SOLVE_NEXT_BLOCK_AT = 1;
constexpr uint32_t MIN_POINTS = SOLVE_NEXT_BLOCK_AT + 1;

// Tile order of the constants CB, written once by the reader and never popped:
// M_off, M_ref, M_extent of each group, then one E_j per lane in use.
constexpr uint32_t CONSTS_PER_GROUP = 3;
constexpr uint32_t const_m_off(uint32_t g) { return CONSTS_PER_GROUP * g; }
constexpr uint32_t const_m_ref(uint32_t g) { return CONSTS_PER_GROUP * g + 1; }
constexpr uint32_t const_m_extent(uint32_t g) { return CONSTS_PER_GROUP * g + 2; }
constexpr uint32_t e_tiles(uint32_t points) { return points < LANES ? points : LANES; }
constexpr uint32_t const_e(uint32_t points, uint32_t j) { return CONSTS_PER_GROUP * groups(points) + j; }
constexpr uint32_t const_tiles(uint32_t points) { return const_e(points, e_tiles(points)); }

// One block's staged operands, row-major, tilized by compute: row q =
// [offsets of group 0 (64 B) | ... | attention weights (up to 64 B)], one tile
// per 64 B. Every field starts on a 64 B boundary, so a NoC read lands on one
// from DRAM as well as L1.
constexpr uint32_t BF16_NBYTES = 2;
constexpr uint32_t TILE_ROW_NBYTES = 32 * BF16_NBYTES;
static_assert(LANES * 2 * BF16_NBYTES == TILE_ROW_NBYTES, "a group's offsets must fill exactly one tile row");
static_assert(MAX_POINTS * BF16_NBYTES == TILE_ROW_NBYTES, "a block's attention weights must fit one tile row");
constexpr uint32_t off_tile(uint32_t g) { return g; }
constexpr uint32_t attn_tile(uint32_t points) { return groups(points); }
constexpr uint32_t rm_tiles(uint32_t points) { return groups(points) + 1; }
constexpr uint32_t rm_attn_offset(uint32_t points) { return groups(points) * TILE_ROW_NBYTES; }
constexpr uint32_t rm_row_nbytes(uint32_t points) { return rm_tiles(points) * TILE_ROW_NBYTES; }

// A float32 reference tile enters the FPU as this many bf16 tiles, whose sum is
// the point exactly (8 + 8 + 8 significant bits).
constexpr uint32_t FP32_REF_PARTS = 3;

// CB depths, in pages. The reader stages block t+1 only after block t's corner
// tiles arrive, and compute pops block t's staged page before it pushes block
// t's corner tiles, so one block of staging is enough. The reader holds block
// t's corner tiles for the whole block while compute solves block t+1, and
// compute solves block t+1 while still extracting block t's weights, so those
// hold two blocks. The reference tile is one page so that a block whose
// reference points did not change can push it again without rewriting it.
constexpr uint32_t WEIGHTS_PER_GROUP = 4;  // w_NW, w_NE, w_SW, w_SE
constexpr uint32_t weight_tiles(uint32_t points) { return WEIGHTS_PER_GROUP * groups(points); }
constexpr uint32_t weight_cb_pages(uint32_t points) { return 2 * weight_tiles(points); }
constexpr uint32_t corner_cb_pages(uint32_t points) { return 2 * groups(points); }
constexpr uint32_t rm_cb_pages(uint32_t points) { return rm_tiles(points); }
constexpr uint32_t REF_TILE_CB_PAGES = 1;

// Per-level constants as fp32 bit patterns in the runtime args: (primary,
// secondary, bias, extent) for x then y. Compute reads them on every path; the
// reader writes them into M_off / M_ref / M_extent on the tiled one.
constexpr uint32_t CONSTANTS_PER_AXIS = 4;
constexpr uint32_t CONSTANTS_PER_LEVEL = 2 * CONSTANTS_PER_AXIS;

}  // namespace fused_msda_tiled_geometry
