// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_sfpu.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"

template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(Callable&& sfpu_func, std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(dst_index);

    _llk_math_eltwise_sfpu_apply_vector_mode_<TILE_SHAPE>(std::forward<Callable>(sfpu_func), vector_mode, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a unary SFPU functor over the tile described by TENSOR_SHAPE.
 *
 * Calls sfpu_func(args...) once per face of the tile; see @ref _llk_math_eltwise_sfpu_for_each_face_.
 *
 * @tparam TENSOR_SHAPE: Tile to process, default = full 32x32 tile
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address and the face stride
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor that processes one face per call
 * @param dst_index: Dest tile index to process
 * @param args: Extra arguments passed to every sfpu_func call
 */
template <
    TensorShape TENSOR_SHAPE                = DEFAULT_TENSOR_SHAPE,
    ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32,
    typename Callable,
    typename... Args>
inline void _llk_math_eltwise_unary_sfpu_run_(Callable&& sfpu_func, std::uint32_t dst_index, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(dst_index);
    _llk_math_eltwise_sfpu_for_each_face_<TENSOR_SHAPE, TILE_SHAPE>(sfpu_func, args...);
    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a unary SFPU functor that walks Dest itself, calling it exactly once.
 *
 * For functors that do their own face iteration (the legacy VectorMode::None / RC_custom callers).
 *
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor, called once as sfpu_func(args...)
 * @param dst_index: Dest tile index to process
 * @param args: Extra arguments passed to sfpu_func
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_run_once_(Callable&& sfpu_func, std::uint32_t dst_index, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(dst_index);
    std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    _llk_math_eltwise_sfpu_done_();
}
