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
