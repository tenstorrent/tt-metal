// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"
#include "sfpu/walk.h"
#include "tensor_shape.h"

using namespace ckernel;

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_sfpu_done_()
{
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

template <DstSync Dst>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
}

/**
 * @brief Dest tile region a legacy VectorMode walks.
 *
 * RC is the whole tile, R the top row of faces (0, 1) and C the left column of faces (0, 2).
 * None and RC_custom select no region: the functor is called once and walks Dest itself.
 *
 * @tparam VECTOR_MODE: Legacy face selection, values = <RC/R/C>
 */
template <VectorMode VECTOR_MODE>
constexpr TensorShape _llk_math_eltwise_sfpu_vector_mode_tensor_shape_()
{
    static_assert(
        VECTOR_MODE == VectorMode::RC || VECTOR_MODE == VectorMode::R || VECTOR_MODE == VectorMode::C,
        "only RC, R and C select a Dest region; None and RC_custom call the functor once");

    if constexpr (VECTOR_MODE == VectorMode::RC)
    {
        return DEFAULT_TENSOR_SHAPE;
    }
    else if constexpr (VECTOR_MODE == VectorMode::R)
    {
        return make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, 1, MAX_NUM_FACES_C_DIM);
    }
    else
    {
        return make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, 1);
    }
}

/// @ref _llk_math_eltwise_sfpu_vector_mode_tensor_shape_ with static storage, so it can be a const TensorShape& template argument.
template <VectorMode VECTOR_MODE>
inline constexpr TensorShape vector_mode_tensor_shape = _llk_math_eltwise_sfpu_vector_mode_tensor_shape_<VECTOR_MODE>();

/**
 * @brief Legacy VectorMode dispatch, kept as a shim over llk::sfpu::for_each_face.
 *
 * RC, R and C walk the region @ref vector_mode_tensor_shape maps them to. Any other mode (None,
 * RC_custom) calls sfpu_func once, for functors that walk Dest themselves.
 *
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types passed to sfpu_func
 * @param sfpu_func: SFPU functor; processes one face per call for RC/R/C
 * @param vector_mode: Legacy face selection, values = <RC/R/C/None/RC_custom>
 * @param args: Arguments passed to sfpu_func
 */
template <typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    const auto face = [&]() __attribute__((always_inline)) { sfpu_func(args...); };

    if (vector_mode == VectorMode::RC)
    {
        llk::sfpu::for_each_face<vector_mode_tensor_shape<VectorMode::RC>>(face);
    }
    else if (vector_mode == VectorMode::R)
    {
        llk::sfpu::for_each_face<vector_mode_tensor_shape<VectorMode::R>>(face);
    }
    else if (vector_mode == VectorMode::C)
    {
        llk::sfpu::for_each_face<vector_mode_tensor_shape<VectorMode::C>>(face);
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}
