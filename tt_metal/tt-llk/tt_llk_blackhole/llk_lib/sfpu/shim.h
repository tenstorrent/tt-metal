// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_sfpu.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"
#include "sfpu/common.h"
#include "sfpu/traverse.h"
#include "tensor_shape.h"

/**
 * Legacy SFPU entry points, kept for callers that have not moved to TensorShape yet.
 */

namespace llk::sfpu::detail
{

/// Dest region a legacy VectorMode walks: RC the whole tile, R faces 0 and 1, C faces 0 and 2.
template <VectorMode Mode>
inline constexpr ckernel::TensorShape vector_mode_shape = []
{
    static_assert(Mode == VectorMode::RC || Mode == VectorMode::R || Mode == VectorMode::C, "only RC, R and C select a Dest region");

    if constexpr (Mode == VectorMode::RC)
    {
        return ckernel::DEFAULT_TENSOR_SHAPE;
    }
    else if constexpr (Mode == VectorMode::R)
    {
        return ckernel::make_tensor_shape(ckernel::MAX_FACE_R_DIM, ckernel::MAX_FACE_C_DIM, 1, ckernel::MAX_NUM_FACES_C_DIM);
    }
    else
    {
        return ckernel::make_tensor_shape(ckernel::MAX_FACE_R_DIM, ckernel::MAX_FACE_C_DIM, ckernel::MAX_NUM_FACES_R_DIM, 1);
    }
}();

} // namespace llk::sfpu::detail

/// Legacy name of @ref llk::sfpu::calculation_start.
inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    llk::sfpu::calculation_start(dst_index);
}

/// Legacy name of @ref llk::sfpu::calculation_end.
inline void _llk_math_eltwise_sfpu_done_()
{
    llk::sfpu::calculation_end();
}
