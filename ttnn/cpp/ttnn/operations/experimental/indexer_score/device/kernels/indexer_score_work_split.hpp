// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Work-split arithmetic for indexer_score, shared by the host factory (maps groups/bands onto the
// core grid) and the kernels (WorkUnitSpan walks each core's (group, band) cells).
// Single-sourcing stops host/device desync (else cores overlap or miss output tiles).
// Pure uint32_t math (no std::, no CT args) so it includes cleanly on both sides.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// Schedule: every q-row-group is dealt the full [0, k_len_tiles) k rectangle (not just its causal
// prefix), so the deal is uniform across groups -- which is what lets the banded multicast apply.
// Future/diagonal tiles are computed and masked to -inf in-band (see valid_prefix_tiles); cost is
// negligible at high sp_rank (~2x only near sp~0).

/** This q-tile-row's causal diagonal tile (the k-tile holding key == query). Normally chunk_start_tiles +
 *  q_row_abs, but on the mid-slab BOUNDARY chip the q-rows straddle a slab boundary: rows at/after
 *  straddle_q_tile jump by straddle_jump_tiles (the global K gap between the two slabs the chip spans).
 *  straddle_jump_tiles == 0 (every non-boundary device, and the chunk-aligned case) leaves it linear. */
constexpr uint32_t causal_diag_tile(
    uint32_t q_row_abs, uint32_t chunk_start_tiles, uint32_t straddle_q_tile, uint32_t straddle_jump_tiles) {
    return chunk_start_tiles + q_row_abs + (q_row_abs >= straddle_q_tile ? straddle_jump_tiles : 0);
}

/** Unmasked prefix k-tiles of q-tile-row q_row_abs in a unit (start k_tile_start, k_tiles_in_unit
 *  valid). Tiles [0, this) are below the diagonal (no mask); diagonal and beyond are masked. */
constexpr uint32_t valid_prefix_tiles(
    uint32_t q_row_abs,
    uint32_t k_tile_start,
    uint32_t k_tiles_in_unit,
    uint32_t chunk_start_tiles,
    uint32_t straddle_q_tile,
    uint32_t straddle_jump_tiles) {
    const uint32_t diag_tile = causal_diag_tile(q_row_abs, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles);
    const uint32_t v = diag_tile > k_tile_start ? diag_tile - k_tile_start : 0;
    return v < k_tiles_in_unit ? v : k_tiles_in_unit;
}

/** Work units per q-row-group: ceil of the full k rectangle over the k chunk. Uniform across groups
 *  under the dense schedule. */
constexpr uint32_t units_in_group(uint32_t k_tiles_per_unit, uint32_t k_len_tiles) {
    return (k_len_tiles + k_tiles_per_unit - 1) / k_tiles_per_unit;
}

// ---- banded-product grid mapping (shared by factory + perf model so their core counts agree) -------
// G groups -> grid rows (q-mcast along a row); U k-bands -> grid columns (k-mcast down a column).

/** Largest divisor of n that is <= cap (always >= 1; n <= cap returns n). */
constexpr uint32_t largest_divisor_leq(uint32_t n, uint32_t cap) {
    uint32_t best = 1;
    for (uint32_t divisor = 2; divisor <= cap && divisor <= n; ++divisor) {
        if (n % divisor == 0) {
            best = divisor;
        }
    }
    return best;
}

/** Grid rows for G groups: the largest divisor of G <= grid_y, so every used row carries the same
 *  G/rows groups. Uniformity keeps a column's k-mcast in lockstep; a prime G > grid_y falls back to 1
 *  row (k-mcast off, still correct). */
constexpr uint32_t rows_for_groups(uint32_t groups, uint32_t grid_y) { return largest_divisor_leq(groups, grid_y); }

/** Grid columns for U k-bands: min(U, grid_x); spare columns idle when U < grid_x. */
constexpr uint32_t cols_for_bands(uint32_t bands, uint32_t grid_x) { return bands < grid_x ? bands : grid_x; }

/** Row-block replication factor: when G groups under-fill the grid rows (short sequences), replicate each
 *  group across this many row-blocks and split its band range across them, filling the idle rows. Capped by
 *  the idle-row count (grid_y / group_rows) and by one band per (block, column) cell (bands / cols), so it is
 *  always >= 1. 1 == the single-band-row schedule (deployed long-sequence cases, where group_rows == grid_y). */
constexpr uint32_t band_row_blocks(uint32_t groups, uint32_t bands, uint32_t grid_x, uint32_t grid_y) {
    const uint32_t fill = grid_y / rows_for_groups(groups, grid_y);  // idle-row fill factor
    const uint32_t deep = bands / cols_for_bands(bands, grid_x);     // never finer than one band per cell
    const uint32_t blocks = fill < deep ? fill : deep;
    return blocks < 1u ? 1u : blocks;
}

/** Cores the banded schedule lights up: (group_rows * row_blocks) rows x cols columns. Single-sourced so the
 *  factory's core rectangle and the perf model's core count cannot drift. */
constexpr uint32_t banded_core_count(uint32_t groups, uint32_t bands, uint32_t grid_x, uint32_t grid_y) {
    return rows_for_groups(groups, grid_y) * band_row_blocks(groups, bands, grid_x, grid_y) *
           cols_for_bands(bands, grid_x);
}

}  // namespace ttnn::operations::experimental::indexer_score
