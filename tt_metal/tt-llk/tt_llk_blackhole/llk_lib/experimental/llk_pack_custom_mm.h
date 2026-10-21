// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "experimental/llk_custom_mm_common.h"

namespace ckernel
{

// Point the packer at custom_mm's output tiles in DEST: the Dense layouts use a 16-bit pack source and 32 rows per
// tile, Compact puts output face 1 at row 8 and tiles 16 rows apart. The Default layouts need no change.
// Only change the face and tile strides; preserve the configured pack formats and geometry.
template <CustomMmDestLayout layout = CustomMmDestLayout::Default, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_custom_mm_init_()
{
    if constexpr (custom_mm_tile_rows(layout) == 32)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
    }
    if constexpr (layout == CustomMmDestLayout::Compact)
    {
        constexpr std::uint32_t row_bytes = FACE_C_DIM * (is_fp32_dest_acc_en ? 4 : 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>((FACE_R_DIM / 2) * row_bytes);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * row_bytes);
    }
}

// Restore the corresponding 16-row face and 64-row tile strides after the final Dense or Compact operation.
template <CustomMmDestLayout layout = CustomMmDestLayout::Default, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_custom_mm_uninit_()
{
    if constexpr (custom_mm_tile_rows(layout) == 32)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
    }
    if constexpr (layout == CustomMmDestLayout::Compact)
    {
        constexpr std::uint32_t row_bytes = FACE_C_DIM * (is_fp32_dest_acc_en ? 4 : 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_R_DIM * row_bytes);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_R_DIM * row_bytes);
    }
}

} // namespace ckernel
