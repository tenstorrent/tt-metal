// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel::isolate_sfpu
{

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

} // namespace ckernel::isolate_sfpu
