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

/**
 * @brief Initializes shared SFPU state for ternary element-wise operations.
 *
 * Programs ADDR_MOD_7 with dest.incr=0 via the common SFPU init. Per-op
 * state (e.g. ADDR_MOD_6 for ops that auto-advance dest) is set up by the
 * op's own @c _init_<op>_ call after this one.
 *
 * @tparam sfpu_op  The ternary SFPU operation type (e.g. @c SfpuType::where).
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    _llk_math_sfpu_init_();
}

/**
 * @brief Dispatches a ternary SFPU kernel over faces selected by @p vector_mode.
 *
 * Sets the DEST section base to tile 0, calls @p sfpu_func once per selected face with the
 * supplied tile indices, advances the face pointer between calls, then signals
 * SFPU done.
 *
 * @tparam TILE_SHAPE     Dest tile shape used when setting the tile-zero destination base.
 * @tparam F              Callable type matching the ternary SFPU kernel signature.
 * @tparam ARGS           Any extra arguments forwarded verbatim to @p sfpu_func.
 *
 * @param sfpu_func       Ternary SFPU kernel (e.g. @c calculate_where<false>).
 * @param dst_index_in0   DEST tile index for the first input operand (e.g. condition).
 * @param dst_index_in1   DEST tile index for the second input operand (e.g. true_val).
 * @param dst_index_in2   DEST tile index for the third input operand (e.g. false_val).
 * @param dst_index_out   DEST tile index that receives the result.
 * @param vector_mode     Faces to process: R (0-1), C (0,2), RC (all 4, default), or scalar (once).
 * @param args            Extra arguments forwarded to @p sfpu_func after the tile indices.
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    VectorMode vector_mode = VectorMode::RC,
    Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    _llk_math_eltwise_sfpu_apply_vector_mode_<TILE_SHAPE>(
        std::forward<Callable>(sfpu_func), vector_mode, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a ternary SFPU functor over the tile described by TENSOR_SHAPE.
 *
 * Calls sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...) once per face of the tile; see @ref
 * _llk_math_eltwise_sfpu_for_each_face_.
 *
 * @tparam TENSOR_SHAPE: Tile to process, default = full 32x32 tile
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address and the face stride
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor that processes one face per call
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_in2: Dest tile index of the third input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to every sfpu_func call
 */
template <
    TensorShape TENSOR_SHAPE                = DEFAULT_TENSOR_SHAPE,
    ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32,
    typename Callable,
    typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_run_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    _llk_math_eltwise_sfpu_for_each_face_<TENSOR_SHAPE, TILE_SHAPE>(sfpu_func, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a ternary SFPU functor that walks Dest itself, calling it exactly once.
 *
 * For functors that do their own face iteration (the legacy VectorMode::None / RC_custom callers).
 *
 * @tparam TILE_SHAPE: Dest slot shape, used for the start address
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor, called once as sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...)
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_in2: Dest tile index of the third input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to sfpu_func
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_run_once_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_<TILE_SHAPE>(0);
    std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    _llk_math_eltwise_sfpu_done_();
}

namespace ckernel::sfpu
{

/**
 * @brief CRTP base for ternary SFPU op classes; provides run().
 *
 * run() checks the Dest indices, then calls Op::calculate(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...) once per face of the tile,
 * or once in total when Op::walks_faces is false.
 *
 * @tparam Op: The derived op class
 * @tparam SLOT: Dest slot shape the tiles live in, used for the start address, face stride and bounds check
 */
template <typename Op, ckernel::trisc::DstTileShape SLOT = ckernel::trisc::DstTileShape::Tile32x32>
struct SfpuTernaryOp : SfpuOpBase<Op>
{
    static_assert(_llk_math_eltwise_sfpu_slot_faces_c_<SLOT>() > 0, "SFPU op classes need a Dest slot at least 32x16");

    /**
     * @brief Run Op on the whole Dest slot (a full 32x32 tile in the default slot).
     *
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to Op::calculate
     */
    template <typename... Args>
    static inline __attribute__((always_inline)) void run(
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        run<_llk_math_eltwise_sfpu_slot_tensor_shape_<SLOT>()>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /**
     * @brief Run Op on the tile described by TENSOR_SHAPE.
     *
     * @tparam TENSOR_SHAPE: Tile to process; must fit in SLOT, see @ref _llk_math_eltwise_sfpu_for_each_face_
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to Op::calculate
     */
    template <TensorShape TENSOR_SHAPE, typename... Args>
    static inline __attribute__((always_inline)) void run(
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_out);
        if constexpr (Op::walks_faces)
        {
            _llk_math_eltwise_ternary_sfpu_run_<TENSOR_SHAPE, SLOT>(Op::calculate, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
        }
        else
        {
            static_assert(_llk_math_eltwise_sfpu_is_full_tile_<SLOT>(TENSOR_SHAPE), "An op that walks Dest itself processes the full tile");
            _llk_math_eltwise_ternary_sfpu_run_once_<SLOT>(Op::calculate, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
        }
    }

    /**
     * @brief Bridge for compute APIs that still take a VectorMode; new code uses run<TENSOR_SHAPE>().
     *
     * RC runs the whole Dest slot, R its top row of faces, C its left column of faces, and None or
     * RC_custom a single face; in a 32x32 slot that is a 32x32, 16x32, 32x16 and 16x16 tile.
     * An op whose calculate() walks Dest itself ignores vector_mode.
     *
     * @param vector_mode: Legacy face selection, values = <RC/R/C/None/RC_custom>
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to Op::calculate
     */
    template <typename... Args>
    static inline __attribute__((always_inline)) void run_vector_mode(
        const VectorMode vector_mode,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        if constexpr (!Op::walks_faces)
        {
            run(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else if (vector_mode == VectorMode::RC)
        {
            run<_llk_math_eltwise_sfpu_slot_tensor_shape_<SLOT>()>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else if (vector_mode == VectorMode::R)
        {
            run<make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, 1, _llk_math_eltwise_sfpu_slot_faces_c_<SLOT>())>(
                dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else if (vector_mode == VectorMode::C)
        {
            run<make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, 1)>(
                dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else
        {
            run<tensor_shape_from_tile_dims(MAX_FACE_R_DIM, MAX_FACE_C_DIM)>(
                dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
    }
};

/**
 * @brief Ternary SFPU adaptor for ad-hoc functors.
 *
 * Runs a functor passed at the call site, for kernels that have no op class (local kernels in ttnn,
 * models and tt-train). The functor is called as sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...).
 */
struct TernaryFn
{
    /**
     * @brief Run a per-face functor on the full 32x32 tile.
     *
     * @param sfpu_func: SFPU functor that processes one face per call
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to sfpu_func
     */
    template <typename Callable, typename... Args>
    static inline __attribute__((always_inline)) void run(
        Callable&& sfpu_func,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        run<DEFAULT_TENSOR_SHAPE>(std::forward<Callable>(sfpu_func), dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /**
     * @brief Run a per-face functor on the tile described by TENSOR_SHAPE.
     *
     * @tparam TENSOR_SHAPE: Tile to process; must fit in SLOT, see @ref _llk_math_eltwise_sfpu_for_each_face_
     * @tparam SLOT: Dest slot shape the tiles live in
     * @param sfpu_func: SFPU functor that processes one face per call
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to sfpu_func
     */
    template <TensorShape TENSOR_SHAPE, ckernel::trisc::DstTileShape SLOT = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
    static inline __attribute__((always_inline)) void run(
        Callable&& sfpu_func,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_out);
        _llk_math_eltwise_ternary_sfpu_run_<TENSOR_SHAPE, SLOT>(
            std::forward<Callable>(sfpu_func), dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /**
     * @brief Run a functor that walks Dest itself, calling it exactly once.
     *
     * @tparam SLOT: Dest slot shape the tiles live in
     * @param sfpu_func: SFPU functor
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to sfpu_func
     */
    template <ckernel::trisc::DstTileShape SLOT = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
    static inline __attribute__((always_inline)) void run_once(
        Callable&& sfpu_func,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_<SLOT>(dst_index_out);
        _llk_math_eltwise_ternary_sfpu_run_once_<SLOT>(
            std::forward<Callable>(sfpu_func), dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /// Op-agnostic SFPU init, for functors that need no per-op state.
    static inline __attribute__((always_inline)) void init()
    {
        _llk_math_eltwise_sfpu_init_();
    }

    /**
     * @brief Op-agnostic SFPU init followed by an init functor.
     *
     * @param init_func: Per-op init to run after the op-agnostic init
     * @param args: Arguments passed to init_func
     */
    template <typename InitCallable, typename... Args>
    static inline __attribute__((always_inline)) void init(InitCallable&& init_func, Args&&... args)
    {
        _llk_math_eltwise_sfpu_init_();
        std::forward<InitCallable>(init_func)(std::forward<Args>(args)...);
    }
};

} // namespace ckernel::sfpu
