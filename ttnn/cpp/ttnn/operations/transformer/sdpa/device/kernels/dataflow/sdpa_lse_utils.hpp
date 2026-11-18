// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Log-sum-exp output for SDPA writers (return_lse). The compute kernel hands the writer, per Q row, the
// running max m of the raw scores q.k and the softmax denominator l = sum(exp(scale * (q.k - m))), each in
// column 0 of a stats tile. The writer forms lse = scale * m + log(l) (the log of the softmax denominator in
// the kernel's logit space scale * q.k) in fp32 on the RISC-V (sdpa_lse_math.hpp) and writes it to an fp32
// TILE tensor, one tile row per Q row: the value in column 0, zeros in columns 1..31 (the writer zero-fills
// its staging tile).
//
// The SFPU would round lse to the 16-bit DEST format when fp32_dest_acc_en is off; lse values reach the
// tens, so bf16 rounding (up to 0.125) would dominate the merge error.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "sdpa_lse_math.hpp"

namespace sdpa_lse {

// Element (row, 0) of a stats tile at L1 address tile_addr, as fp32 bits. Row r of a 32x32 tile is in face
// 0 (r < 16) or face 2 (r >= 16); a 16x32 half tile only has rows 0-15. fp32 selects a Float32 stats tile,
// otherwise Float16_b.
template <bool fp32>
FORCE_INLINE uint32_t stats_col0_bits(uint32_t tile_addr, uint32_t row) {
    const uint32_t idx = ((row >> 4) * 512) + ((row & 15u) * 16);
    if constexpr (fp32) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[idx];
    } else {
        return static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr)[idx]) << 16;
    }
}

// Write value_bits to element (row, 0) of the fp32 32x32 tile at L1 address tile_addr (face 0 or 2).
FORCE_INLINE void set_fp32_tile_col0(uint32_t tile_addr, uint32_t row, uint32_t value_bits) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[((row >> 4) * 512) + ((row & 15u) * 16)] = value_bits;
}

// Fill column 0 of the fp32 tile at lse_tile_addr (the other columns stay as the caller zero-filled them)
// from the matching m / l stats tiles: rows [0, stats_rows) get scale * m + log(l) (or kNoKeyLseBits when
// no_key_row(row) holds), the rest kNoKeyLseBits.
template <bool l_fp32, typename NoKeyFn>
inline void fill_lse_tile(
    uint32_t lse_tile_addr,
    uint32_t m_tile_addr,
    uint32_t l_tile_addr,
    uint32_t stats_rows,
    uint32_t scale_bits,
    const NoKeyFn& no_key_row) {
    for (uint32_t row = 0; row < 32; ++row) {
        uint32_t value = kNoKeyLseBits;
        if (row < stats_rows && !no_key_row(row)) {
            value = lse_bits(
                stats_col0_bits<false>(m_tile_addr, row), stats_col0_bits<l_fp32>(l_tile_addr, row), scale_bits);
        }
        set_fp32_tile_col0(lse_tile_addr, row, value);
    }
}

// fill_lse_tile for one of the candidates a packed spec-decode stats tile holds: rows [0, rows) of the lse tile from
// stats rows [first_row, first_row + rows) (kNoKeyLseBits for all of them when no_key), the rest kNoKeyLseBits.
template <bool l_fp32>
inline void fill_lse_tile_rows(
    uint32_t lse_tile_addr,
    uint32_t m_tile_addr,
    uint32_t l_tile_addr,
    uint32_t first_row,
    uint32_t rows,
    uint32_t scale_bits,
    bool no_key) {
    for (uint32_t row = 0; row < 32; ++row) {
        uint32_t value = kNoKeyLseBits;
        if (row < rows && !no_key) {
            value = lse_bits(
                stats_col0_bits<false>(m_tile_addr, first_row + row),
                stats_col0_bits<l_fp32>(l_tile_addr, first_row + row),
                scale_bits);
        }
        set_fp32_tile_col0(lse_tile_addr, row, value);
    }
}

// Fill column 0 of the (zero-filled) fp32 tile at lse_tile_addr with kNoKeyLseBits.
FORCE_INLINE void fill_no_key_lse_tile(uint32_t lse_tile_addr) {
    for (uint32_t row = 0; row < 32; ++row) {
        set_fp32_tile_col0(lse_tile_addr, row, kNoKeyLseBits);
    }
}

}  // namespace sdpa_lse
