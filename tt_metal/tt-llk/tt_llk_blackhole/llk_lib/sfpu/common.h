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
    // todo(sstanisic) : refactor to DESTTILE
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
}

/// Start an SFPU calculation on Dest tile @p dst_index: point Dest at it and wait for the math pipe to drain.
inline void calculation_start(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

/// End an SFPU calculation: reset the Dest address.
inline void calculation_end()
{
    math::clear_dst_reg_addr();
}

/// RAII pair of @ref calculation_start and @ref calculation_end.
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
