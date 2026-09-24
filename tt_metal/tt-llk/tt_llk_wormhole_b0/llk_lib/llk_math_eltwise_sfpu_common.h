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
#include "tensor_shape.h"

using namespace ckernel;

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_sfpu_done_()
{
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
}

template <DstSync Dst>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
}

/**
 * @brief Program the SFPU state shared by every SFPU op: the SFPU config register and ADDR_MOD_7.
 *
 * ADDR_MOD_7 is programmed with all increments zero. SFPU kernels typically run alongside A2D, which
 * uses ADDR_MOD_0 and ADDR_MOD_2, so this slot does not conflict.
 */
inline void _llk_math_eltwise_sfpu_configure_common_()
{
    sfpu::_init_sfpu_config_reg();

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Op-agnostic SFPU init: the shared SFPU state plus a counter reset.
 *
 * Programs no ADDR_MOD_6. Ops that need it program it in their own init, which runs after this one.
 */
inline void _llk_math_eltwise_sfpu_init_()
{
    _llk_math_eltwise_sfpu_configure_common_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Advance the SFPU Dest address past NUM_FACES faces without processing them.
 *
 * @tparam NUM_FACES: Number of faces to skip, expanded at compile time into straight-line increments
 */
template <int NUM_FACES>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_skip_faces_()
{
    if constexpr (NUM_FACES > 0)
    {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_skip_faces_<NUM_FACES - 1>();
    }
}

/**
 * @brief Call an SFPU functor once per face of the tile described by TENSOR_SHAPE.
 *
 * The tile starts at face 0 of the current 32x32 Dest tile. Faces are visited in row-major order and
 * faces outside the shape are skipped, so a 16x32 tile visits faces 0 and 1 and a 32x16 tile visits
 * faces 0 and 2. The walk always ends at the end of the Dest tile.
 *
 * The static_asserts limit TENSOR_SHAPE to full-face tiles of at most 2x2 faces, all of which are in the
 * math TensorShape coverage table, so no runtime LLK_VALIDATE_TENSOR_SHAPE_MATH check is needed.
 *
 * @tparam TENSOR_SHAPE: Tile to process; must be made of full 16x16 faces, at most 2x2 of them
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor to run on each face; processes one face per call
 * @param args: Arguments passed to every sfpu_func call
 */
template <TensorShape TENSOR_SHAPE, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_for_each_face_(Callable&& sfpu_func, Args&&... args)
{
    static_assert(TENSOR_SHAPE.face_r_dim == MAX_FACE_R_DIM && TENSOR_SHAPE.face_c_dim == MAX_FACE_C_DIM, "SFPU face walk requires full 16x16 faces");
    static_assert(TENSOR_SHAPE.num_faces_r_dim >= 1 && TENSOR_SHAPE.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "SFPU face walk supports 1 or 2 rows of faces");
    static_assert(TENSOR_SHAPE.num_faces_c_dim >= 1 && TENSOR_SHAPE.num_faces_c_dim <= MAX_NUM_FACES_C_DIM, "SFPU face walk supports 1 or 2 columns of faces");

    constexpr int FACES_R = TENSOR_SHAPE.num_faces_r_dim;
    constexpr int FACES_C = TENSOR_SHAPE.num_faces_c_dim;

    if constexpr (FACES_C == MAX_NUM_FACES_C_DIM)
    {
        // Whole rows of faces are contiguous in Dest.
#pragma GCC unroll 0
        for (int face = 0; face < FACES_R * FACES_C; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        // One face per row of faces; skip the rest of the row.
#pragma GCC unroll 0
        for (int face_r = 0; face_r < FACES_R; face_r++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_skip_faces_<MAX_NUM_FACES_C_DIM - FACES_C>();
        }
    }

    // Skip the rows of faces outside the shape.
    _llk_math_eltwise_sfpu_skip_faces_<(MAX_NUM_FACES_R_DIM - FACES_R) * MAX_NUM_FACES_C_DIM>();
}

/**
 * @brief Legacy VectorMode dispatch, kept as a shim over @ref _llk_math_eltwise_sfpu_for_each_face_.
 *
 * RC, R and C walk a 32x32, 16x32 and 32x16 tile respectively. Any other mode (None, RC_custom)
 * calls sfpu_func once, for functors that walk Dest themselves.
 *
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor to run
 * @param vector_mode: Legacy face selection, values = <RC/R/C/None/RC_custom>
 * @param args: Arguments passed to sfpu_func
 */
template <typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    if (vector_mode == VectorMode::RC)
    {
        _llk_math_eltwise_sfpu_for_each_face_<DEFAULT_TENSOR_SHAPE>(sfpu_func, args...);
    }
    else if (vector_mode == VectorMode::R)
    {
        _llk_math_eltwise_sfpu_for_each_face_<tensor_shape_from_tile_dims(MAX_FACE_R_DIM, MAX_TILE_C_DIM)>(sfpu_func, args...);
    }
    else if (vector_mode == VectorMode::C)
    {
        _llk_math_eltwise_sfpu_for_each_face_<tensor_shape_from_tile_dims(MAX_TILE_R_DIM, MAX_FACE_C_DIM)>(sfpu_func, args...);
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}

/**
 * @brief Assert that a Dest tile index fits in Dest for the kernel's sync mode.
 *
 * Reads the kernel-global DST_SYNC_MODE; the accumulation mode is read at runtime by
 * get_dest_max_tiles_rt. A build that has not defined DST_SYNC_MODE before including this header
 * (standalone tt-llk tests that call the LLK directly) skips the check.
 *
 * @param dst_index: Dest tile index to check
 */
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_check_dst_index_([[maybe_unused]] const std::uint32_t dst_index)
{
#ifdef DST_SYNC_MODE
    _llk_math_eltwise_sfpu_assert_dst_index_<DST_SYNC_MODE>(dst_index, "dst_index exceeds max dest tiles");
#endif
}

/**
 * @brief Whether a TensorShape is the full 32x32 tile.
 *
 * @param tensor_shape: Shape to test
 */
constexpr bool _llk_math_eltwise_sfpu_is_full_tile_(const TensorShape tensor_shape)
{
    return tensor_shape.face_r_dim == DEFAULT_TENSOR_SHAPE.face_r_dim && tensor_shape.face_c_dim == DEFAULT_TENSOR_SHAPE.face_c_dim &&
           tensor_shape.num_faces_r_dim == DEFAULT_TENSOR_SHAPE.num_faces_r_dim && tensor_shape.num_faces_c_dim == DEFAULT_TENSOR_SHAPE.num_faces_c_dim;
}

namespace ckernel::sfpu
{

/**
 * @brief CRTP base shared by every SFPU op class.
 *
 * An op class derives from one of the arity bases (SfpuUnaryOp, SfpuBinaryOp, SfpuTernaryOp) and
 * supplies a static calculate() that delegates to its ckernel. It overrides init_op() when it needs
 * per-op state (e.g. ADDR_MOD_6 or programmable constants), and sets walks_faces = false when
 * calculate() walks the whole tile itself instead of processing one face per call.
 *
 * @tparam Op: The derived op class
 */
template <typename Op>
struct SfpuOpBase
{
    /// calculate() processes one face per call. Set to false in ops whose calculate() walks Dest itself.
    static constexpr bool walks_faces = true;

    /**
     * @brief Initialize the SFPU for Op: the op-agnostic SFPU init, then Op::init_op(args...).
     *
     * @param args: Arguments forwarded to Op::init_op
     */
    template <typename... Args>
    static inline __attribute__((always_inline)) void init(Args&&... args)
    {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op(std::forward<Args>(args)...);
    }

    /// Default per-op init: no state beyond the op-agnostic SFPU init.
    static inline __attribute__((always_inline)) void init_op()
    {
    }
};

} // namespace ckernel::sfpu
