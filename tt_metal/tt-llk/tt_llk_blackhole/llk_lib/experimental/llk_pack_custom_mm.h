// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel
{

// DEST rows between consecutive output tiles: the default tile layout and custom_mm's dense layout.
constexpr std::uint32_t CUSTOM_MM_DEFAULT_TILE_ROWS = TILE_NUM_FACES * FACE_R_DIM;
constexpr std::uint32_t CUSTOM_MM_DENSE_TILE_ROWS   = (TILE_NUM_FACES / 2) * FACE_R_DIM;

/**
 * @brief Packer tile-to-tile (W) stride for `rows` DEST rows.
 *
 * The packer counts its strides in bytes of its source format, which is how @ref set_packer_strides programs the
 * default: a row is FACE_C_DIM datums of pack_src_format. The DEST accumulation width is not what matters: a
 * Float16_b output under fp32 accumulation keeps a 2-byte pack source.
 *
 * @param rows: DEST rows between consecutive tiles.
 * @param pack_src_format: The packer's configured source format.
 */
constexpr std::uint32_t custom_mm_pack_w_stride(const std::uint32_t rows, const std::uint32_t pack_src_format)
{
    return rows * FACE_C_DIM * datum_size_in_bytes(pack_src_format);
}

/**
 * @brief Program the packer W stride for custom_mm's dense layout, 32 DEST rows per tile.
 *
 * Only the tile stride changes; the configured pack formats and geometry are preserved.
 *
 * @tparam dense_packing: Lay output tiles 32 DEST rows apart instead of 64; no-op when false.
 * @param pack_src_format: The packer's configured source format for the output operand.
 * @note Call after the pack init that programmed the default strides, with the same dense_packing as
 *       @ref _llk_math_custom_mm_init_ on the math thread, and @ref _llk_pack_custom_mm_uninit_ with the same
 *       arguments after the last dense pack.
 */
template <bool dense_packing = false>
inline void _llk_pack_custom_mm_init_(const std::uint32_t pack_src_format)
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(custom_mm_pack_w_stride(CUSTOM_MM_DENSE_TILE_ROWS, pack_src_format));
    }
}

/**
 * @brief Restore the default 64-row packer W stride after the last dense custom_mm pack.
 *
 * @tparam dense_packing: Must match the preceding @ref _llk_pack_custom_mm_init_; no-op when false.
 * @param pack_src_format: The packer's configured source format for the output operand.
 * @note Call after the last pack of a dense custom_mm sequence and before the next op's pack init.
 */
template <bool dense_packing = false>
inline void _llk_pack_custom_mm_uninit_(const std::uint32_t pack_src_format)
{
    if constexpr (dense_packing)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(custom_mm_pack_w_stride(CUSTOM_MM_DEFAULT_TILE_ROWS, pack_src_format));
    }
}

} // namespace ckernel
