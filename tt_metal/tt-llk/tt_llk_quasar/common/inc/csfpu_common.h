// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::isolate_sfpu
{

constexpr static std::uint32_t TRISC_ID = 3;

// SFPU register-file base addresses: dest region vs SrcS (used by SFPU load/store)
constexpr static unsigned int SFPU_DEST_BASE_ADDR = 0x0;
constexpr static unsigned int SFPU_SRCS_BASE_ADDR = 0x400;

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
            return SFPU_DEST_BASE_ADDR + slot * ydim;
        }
        else
        {
            return SFPU_SRCS_BASE_ADDR + slot * ydim;
        }
    }
};

// SrcS register tile geometry (HW-defined for Quasar SrcS).
// A 32x32 tile is produced/consumed across SLICE_COUNT SrcS slices, where one slice
// holds XDIM * YDIM * ZDIM datums. YDIM halves in 32-bit element mode because the
// SrcS columns are 16-bit wide in HW.
struct srcs_dims
{
    static constexpr std::uint32_t XDIM      = 16; // datums per row of SrcS slice
    static constexpr std::uint32_t ZDIM      = 1;
    static constexpr std::uint32_t YDIM_BASE = 8; // rows per slice when SrcS is in 16-bit mode

    static constexpr std::uint32_t ydim(bool srcs_32bit_mode)
    { // TODO for metal bringup: make programmable based on tensor_shape for tiny tile support
        return srcs_32bit_mode ? (YDIM_BASE / 2) : YDIM_BASE;
    }

    static constexpr std::uint32_t slice_count(bool srcs_32bit_mode)
    {
        return (ckernel::TILE_R_DIM * ckernel::TILE_C_DIM) / (XDIM * ydim(srcs_32bit_mode) * ZDIM);
    }
};

// SrcS runs in 32-bit element mode when the UNP_S destination format is 32-bit wide.
inline constexpr bool _is_srcs_32bit_mode_(const DataFormat unpack_S_dst_format)
{
    return unpack_S_dst_format == DataFormat::Float32 || unpack_S_dst_format == DataFormat::Int32;
}

} // namespace ckernel::isolate_sfpu
