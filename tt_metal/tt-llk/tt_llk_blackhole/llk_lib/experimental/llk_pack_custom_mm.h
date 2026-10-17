// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel
{

// Packer tile-to-tile (W) stride for `rows` DEST rows. The DEST row is 16 datums of the
// packer's source width: 4 bytes under 32-bit dest accumulation, 2 bytes otherwise -- the
// same rule set_packer_strides applies through datum_size_in_bytes(pack_src_format).
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t custom_mm_pack_w_stride(const std::uint32_t rows)
{
    return rows * FACE_C_DIM * (is_fp32_dest_acc_en ? 4 : 2);
}

// Custom matmul's dense layout places consecutive output tiles 32 DEST rows apart.
// Only change the tile stride; preserve the configured pack formats and geometry.
template <bool dense_packing, bool is_fp32_dest_acc_en>
inline void _llk_pack_custom_mm_init_()
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(custom_mm_pack_w_stride<is_fp32_dest_acc_en>((TILE_NUM_FACES / 2) * FACE_R_DIM));
    }
}

// Restore the corresponding 64-row tile stride after the final dense operation.
template <bool dense_packing, bool is_fp32_dest_acc_en>
inline void _llk_pack_custom_mm_uninit_()
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(custom_mm_pack_w_stride<is_fp32_dest_acc_en>(TILE_NUM_FACES * FACE_R_DIM));
    }
}

} // namespace ckernel
