// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"

template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    VectorMode vector_mode = VectorMode::RC,
    Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    _llk_math_eltwise_sfpu_apply_vector_mode_<TILE_SHAPE>(
        std::forward<Callable>(sfpu_func), vector_mode, dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a binary SFPU functor over the tile described by TENSOR_SHAPE.
 *
 * Calls sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, args...) once per face of the tile; see @ref _llk_math_eltwise_sfpu_for_each_face_.
 *
 * @tparam TENSOR_SHAPE: Tile to process, default = full 32x32 tile
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address and the face stride
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor that processes one face per call
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to every sfpu_func call
 */
template <
    TensorShape TENSOR_SHAPE                = DEFAULT_TENSOR_SHAPE,
    ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32,
    typename Callable,
    typename... Args>
inline void _llk_math_eltwise_binary_sfpu_run_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    _llk_math_eltwise_sfpu_for_each_face_<TENSOR_SHAPE, TILE_SHAPE>(sfpu_func, dst_index_in0, dst_index_in1, dst_index_out, args...);
    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a binary SFPU functor that walks Dest itself, calling it exactly once.
 *
 * For functors that do their own face iteration (the legacy VectorMode::None / RC_custom callers).
 *
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor, called once as sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, args...)
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to sfpu_func
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_run_once_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
    _llk_math_eltwise_sfpu_done_();
}
