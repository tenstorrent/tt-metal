// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Balanced M-row band partition. Splits `total_rows` into `num_bands` contiguous bands
// whose heights differ by at most one row, spreading the remainder across the leading
// bands (13 rows / 4 -> 4,3,3,3, not the front-loaded 4,4,4,1). Even heights keep the
// per-band NOC packets uniform, and for num_bands <= total_rows every band is non-empty,
// so the AG writer fires exactly num_bands matmul-aggregator incs and never desyncs from
// the receiver (which always expects num_bands).
//
// The AG writer/reader (strided_all_gather_common.hpp) and the fused matmul in0 reader
// (dm_in0_sender.cpp) both call this so their band boundaries are guaranteed identical.
inline void balanced_band(uint32_t total_rows, uint32_t num_bands, uint32_t band, uint32_t& band_lo, uint32_t& band_h) {
    uint32_t base = total_rows / num_bands;
    uint32_t rem = total_rows % num_bands;
    band_lo = band * base + (band < rem ? band : rem);
    band_h = base + (band < rem ? 1 : 0);
}

// Interleaved two-NoC output-write row ownership (SPLIT_OUTPUT_WRITE + AGMM_INTERLEAVED_OUTPUT_WRITE).
// An output block's M-rows are split between the NOC_1 writer (dm_in1, out_cb_a / c_2) and the NOC_0 writer
// (dm_in0, out_cb_b / c_8). A contiguous [0, split_rows) prefix starves the NOC_0 writer: compute packs
// rows in ascending m, so c_8 stays empty until compute crosses the split boundary and the two writers run
// back-to-back instead of overlapping. Interleaving ownership across the block feeds both writers from the
// first rows so they overlap.
//
// Ratio-preserving Bresenham: row m belongs to NOC_1 iff floor((m+1)*pct/100) > floor(m*pct/100). Summed
// over m in [0, M) the NOC_1 count telescopes to floor(M*pct/100) -- exactly the contiguous split_rows --
// so every CB reserve/push/pop count is unchanged. At pct == 50 this is exact alternation (rows 1,3,5,...
// -> NOC_1); pct == 100 sends every row to NOC_1 (single-writer case).
//
// Contract: compute (copy_block_split / add_bias_block_split) routes each packed row with this predicate,
// and each DM writer iterates all M rows but drains/writes only the rows it owns, in ascending m order. FIFO
// pop order therefore matches compute's push order and the DRAM row index is d0_start + m. All three kernels
// take the same pct from AG_SPLIT_NOC1_PCT, so they never diverge.
inline bool split_row_to_noc1(uint32_t m, uint32_t pct) { return ((m + 1) * pct) / 100 > (m * pct) / 100; }
