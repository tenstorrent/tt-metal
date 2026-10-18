// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel
{

// Custom matmul's dense layout uses a 16-bit pack source and 32 rows per tile.
// Only change the tile stride; preserve the configured pack formats and geometry.
template <bool dense_packing = false>
inline void _llk_pack_custom_mm_init_()
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
    }
}

// Restore the corresponding 64-row tile stride after the final dense operation.
template <bool dense_packing = false>
inline void _llk_pack_custom_mm_uninit_()
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
    }
}

} // namespace ckernel
