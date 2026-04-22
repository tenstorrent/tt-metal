// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::unpack
{
// Number of rows for Unpack functions
constexpr static std::uint32_t UNPACR_STRIDE_MAX_ROWS = 8;
constexpr static std::uint32_t TRISC_ID               = 0;

/**
 * @brief Sets destination register base address for the UNPACK thread (SEC0).
 * Equivalent to ckernel::math::_set_dst_write_addr_ but targets the correct
 * TRISC_ID for the unpack compilation unit.
 * @param tile_index: Tile index in the dest register
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE>
inline void _set_dst_write_addr_(const std::uint32_t tile_index)
{
    const std::uint32_t tile_shape_idx =
        (TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x32) ? 6 : ((TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x16) ? 5 : 4);
    const std::uint32_t dst_index = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

} // namespace ckernel::unpack
