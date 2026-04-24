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

// Provides array-like access to SFPU SrcS slice base addresses.
// Stride between slices = ydim (8 for 16-bit, 4 for 32-bit).
// Usage: SfpuSrcsSlice srcs{PARAM_SRCS_YDIM}; srcs[0] = input0, srcs[1] = input1, srcs[2] = output, etc.
struct SfpuSrcsSlice
{
    const int ydim;

    int operator[](int slice_idx) const
    {
        return SFPU_SRCS_BASE_ADDR + slice_idx * ydim;
    }
};

// Provides array-like access to SFPU Dest slice base addresses within a single tile.
// Tile positioning is handled separately by _set_dst_write_addr_ (dest_section_base);
// this struct only computes the intra-tile slice offset.
// Stride between slices = ydim (8 for 16-bit, 4 for 32-bit).
// Usage: SfpuDestSlice dest{PARAM_SRCS_YDIM}; dest[slice] = offset within tile.
struct SfpuDestSlice
{
    const int ydim;

    int operator[](int slice_idx) const
    {
        return SFPU_DEST_BASE_ADDR + slice_idx * ydim;
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
