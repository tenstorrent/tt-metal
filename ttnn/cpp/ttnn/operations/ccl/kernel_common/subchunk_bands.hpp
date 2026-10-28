// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Balanced M-row band partition
inline void balanced_band(uint32_t total_rows, uint32_t num_bands, uint32_t band, uint32_t& band_lo, uint32_t& band_h) {
    uint32_t base = total_rows / num_bands;
    uint32_t rem = total_rows % num_bands;
    band_lo = band * base + (band < rem ? band : rem);
    band_h = base + (band < rem ? 1 : 0);
}

// Interleaved two-NoC output-write row ownership (SPLIT_OUTPUT_WRITE + AGMM_INTERLEAVED_OUTPUT_WRITE)
inline bool split_row_to_noc1(uint32_t m, uint32_t pct) { return ((m + 1) * pct) / 100 > (m * pct) / 100; }
