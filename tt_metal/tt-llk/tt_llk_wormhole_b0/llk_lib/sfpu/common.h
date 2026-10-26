// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "tensor_shape.h"

namespace llk::sfpu
{

/**
 * @brief Whether a call may compute on Dest lanes outside the region it is given.
 */
enum class PaddingAgnostic
{
    Yes,
    No,
};

/// sfpi::dst_reg index distance between consecutive Dest tiles: a tile is 64 Dest rows.
constexpr std::uint32_t DEST_TILE_STRIDE = (MAX_NUM_FACES * MAX_FACE_R_DIM) / sfpi::SFP_DESTREG_STRIDE;

constexpr std::uint32_t DEST_VECTOR_ROWS     = 4;
constexpr std::uint32_t DEST_VECTOR_ROW_STEP = 1;

constexpr std::uint32_t DEST_VECTOR_COLS     = 8;
constexpr std::uint32_t DEST_VECTOR_COL_STEP = 2;

template <DstSync Dst>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    // todo(sstanisic) : refactor function
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
}

/**
 * @brief Starts an SFPU calculation.
 * @param dst_index The index of the OUTPUT tile in DEST.
 * @note Call before issuing any SFPU INSTRUCTIONS that READ or WRITE to the DEST.
 */
inline void calculation_start(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

/**
 * @brief Ends an SFPU calculation.
 * @note Call after issuing all SFPU INSTRUCTIONS that READ or WRITE to the DEST.
 */
inline void calculation_end()
{
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

/**
 * @brief RAII scope for calculation start and end.
 *
 * Automatically calls calculation_start() on construction and calculation_end() on destruction.
 */
class CalculationScope
{
public:
    CalculationScope(const std::uint32_t dst_index)
    {
        calculation_start(dst_index);
    }

    ~CalculationScope()
    {
        calculation_end();
    }
};

} // namespace llk::sfpu
