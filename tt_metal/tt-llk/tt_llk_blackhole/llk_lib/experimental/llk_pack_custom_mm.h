// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "experimental/llk_custom_mm_common.h"

namespace ckernel
{

// Packer DEST strides are in bytes of the pack source format, not of the DEST width: the packer divides them by the
// pack source datum size (see set_packer_strides). A DEST row is FACE_C_DIM datums.
inline std::uint32_t custom_mm_pack_row_bytes(const std::uint32_t pack_src_format)
{
    return FACE_C_DIM * datum_size_in_bytes(pack_src_format);
}

/**
 * @brief Point the packer at custom_mm's output tiles in DEST.
 *
 * Dense puts tiles 32 rows apart; Compact puts output face 1 at row 8 and tiles 16 rows apart. The Default layouts
 * need no change. Only the face (Z) and tile (W) strides change; the configured pack formats and geometry stay.
 *
 * @tparam layout: DEST layout of the output tiles; must match the math init and execute.
 * @param pack_src_format: The configured pack source format, which sets the stride units.
 * @note Pair with @ref _llk_pack_custom_mm_uninit_ with the same layout and pack source format after the last
 *       custom_mm pack.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_pack_custom_mm_init_(const std::uint32_t pack_src_format)
{
    const std::uint32_t row_bytes = custom_mm_pack_row_bytes(pack_src_format);
    if constexpr (custom_mm_tile_rows(layout) == 32)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(2 * FACE_R_DIM * row_bytes);
    }
    if constexpr (layout == CustomMmDestLayout::Compact)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>((FACE_R_DIM / 2) * row_bytes);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * row_bytes);
    }
}

/**
 * @brief Restore the packer strides that @ref _llk_pack_custom_mm_init_ changed.
 *
 * Dense gets back the 64-row tile stride; Compact also gets back the 16-row face stride. The Default layouts changed
 * nothing.
 *
 * @tparam layout: Must match the init.
 * @param pack_src_format: Must match the init.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_pack_custom_mm_uninit_(const std::uint32_t pack_src_format)
{
    const std::uint32_t row_bytes = custom_mm_pack_row_bytes(pack_src_format);
    if constexpr (layout == CustomMmDestLayout::Compact)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_R_DIM * row_bytes);
    }
    if constexpr (custom_mm_tile_rows(layout) != 64)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_R_DIM * row_bytes);
    }
}

} // namespace ckernel
