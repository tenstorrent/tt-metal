// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel::isolate_sfpu
{

constexpr static std::uint32_t TRISC_ID = 3;

// Identifies which SFPU-addressable register file an operand lives in.
// Used as a compile-time tag for SfpuSlice<REG>
enum class SfpuReg : std::uint8_t
{
    Dest, // SFPU dest region (base = SFPU_DEST_BASE_ADDR == 0)
    SrcS, // SrcS source register (base = SFPU_SRCS_BASE_ADDR)
};

// SfpuSlice<REG> — array-like accessor for SFPU register-file slice base
// addresses inside a single tile, parametrized by which physical register
// the operand lives in (Dest or SrcS).
//
// The meaning of `slot` differs between the two register identities:
//
//   SfpuSlice<SfpuReg::SrcS>:
//     `slot` is the HW-fixed SrcS slot index, set by the unpack/pack
//     engines: in0 = 0, in1 = 1, out = 2.
//
//   SfpuSlice<SfpuReg::Dest>:
//     `slot` is the intra-tile slice index (0, 1, ..., SLICE_COUNT - 1).
//     Tile positioning is handled separately by _set_dst_write_addr_,
//     which programs the dest_section_base; this helper only computes
//     the per-slice offset *within* the tile, so the full Dest address
//     used by the SFPU is
//         dest_section_base + SfpuSlice<SfpuReg::Dest>{ydim}[slice].
//
// Usage:
//   const SfpuSlice<SfpuReg::SrcS> srcs{ydim}; // srcs[0]=in0, srcs[1]=in1, srcs[2]=out
//   const SfpuSlice<SfpuReg::Dest> dest{ydim}; // dest[slice]=intra-tile offset
template <SfpuReg REG>
struct SfpuSlice
{
    const int ydim;

    constexpr int operator[](int slot) const
    {
        if constexpr (REG == SfpuReg::Dest)
        {
            return ckernel::math::SFPU_DEST_BASE_ADDR + slot * ydim;
        }
        else
        {
            return ckernel::math::SFPU_SRCS_BASE_ADDR + slot * ydim;
        }
    }
};

/**
 * @brief Sets destination register base address for the ISOLATE_SFPU thread (SEC3).
 * Equivalent to ckernel::math::_set_dst_write_addr_ but targets the correct
 * TRISC_ID for the ISOLATE_SFPU compilation unit.
 * @param tile_index: Tile index in the dest reg
 * 16bit dest reg data format -> tile_idx = 0 - 7
 * 32bit dest reg data format -> tile_idx = 0 - 3
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE>
inline void _set_dst_write_addr_(const std::uint32_t tile_index)
{
    const std::uint32_t tile_shape_idx =
        (TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x32) ? 6 : ((TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x16) ? 5 : 4);
    const std::uint32_t dst_index = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

inline void _set_dst_write_addr_by_rows_(const std::uint32_t num_rows_per_tile, const std::uint32_t tile_index)
{
    const std::uint32_t tile_shape_idx =
        (num_rows_per_tile == 64)
            ? 6
            : ((num_rows_per_tile == 32) ? 5 : ((num_rows_per_tile == 16) ? 4 : ((num_rows_per_tile == 8) ? 3 : ((num_rows_per_tile == 4) ? 2 : 1))));
    const std::uint32_t dst_index = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

} // namespace ckernel::isolate_sfpu
