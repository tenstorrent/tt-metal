// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/shim.h"

/**
 * @brief Run @p sfpu_func as a complete SFPU calculation, over the Dest region @p vector_mode selects.
 *
 * RC, R and C call @p sfpu_func once per face of their region, through @ref llk::sfpu::walk_faces. None and
 * RC_custom call it once, for functors that walk Dest themselves.
 */
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(Callable&& sfpu_func, std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC, Args&&... args)
{
    const auto face = [&]() __attribute__((always_inline)) { sfpu_func(args...); };

    if (vector_mode == VectorMode::RC)
    {
        llk::sfpu::walk_faces<llk::sfpu::detail::vector_mode_shape<VectorMode::RC>>(dst_index, face);
    }
    else if (vector_mode == VectorMode::R)
    {
        llk::sfpu::walk_faces<llk::sfpu::detail::vector_mode_shape<VectorMode::R>>(dst_index, face);
    }
    else if (vector_mode == VectorMode::C)
    {
        llk::sfpu::walk_faces<llk::sfpu::detail::vector_mode_shape<VectorMode::C>>(dst_index, face);
    }
    else
    {
        llk::sfpu::CalculationScope scope(dst_index);
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}
