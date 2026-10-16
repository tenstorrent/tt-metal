// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_ternary_sfpu.h"

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    VectorMode vector_mode = VectorMode::RC,
    Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(0); // Reuse same sync primitive

    _llk_math_eltwise_sfpu_apply_vector_mode_(
        std::forward<Callable>(sfpu_func), vector_mode, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_(); // Finalize
}

/**
 * @brief Run a ternary SFPU functor over the tile described by TENSOR_SHAPE.
 *
 * Calls sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...) once per face of the tile; see @ref
 * _llk_math_eltwise_sfpu_for_each_face_.
 *
 * @tparam TENSOR_SHAPE: Tile to process, default = full 32x32 tile
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor that processes one face per call
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_in2: Dest tile index of the third input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to every sfpu_func call
 */
template <TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_run_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(0);

    _llk_math_eltwise_sfpu_for_each_face_<TENSOR_SHAPE>(sfpu_func, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);

    _llk_math_eltwise_sfpu_done_();
}

/**
 * @brief Run a ternary SFPU functor that walks Dest itself, calling it exactly once.
 *
 * For functors that do their own face iteration (the legacy VectorMode::None / RC_custom callers).
 *
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor, called once as sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...)
 * @param dst_index_in0: Dest tile index of the first input
 * @param dst_index_in1: Dest tile index of the second input
 * @param dst_index_in2: Dest tile index of the third input
 * @param dst_index_out: Dest tile index that receives the result
 * @param args: Extra arguments passed to sfpu_func
 */
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_run_once_(
    Callable&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(0);

    std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}

namespace ckernel::sfpu
{

/**
 * @brief CRTP base for ternary SFPU op classes; provides run().
 *
 * run() checks the Dest indices, then calls Op::calculate(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...) once per face of
 * the tile, or once in total when Op::walks_faces is false.
 *
 * @tparam Op: The derived op class
 */
template <typename Op>
struct SfpuTernaryOp : SfpuOpBase<Op>
{
    /**
     * @brief Run Op on the full 32x32 tile.
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
        run<DEFAULT_TENSOR_SHAPE>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /**
     * @brief Run Op on the tile described by TENSOR_SHAPE.
     *
     * @tparam TENSOR_SHAPE: Tile to process; see @ref _llk_math_eltwise_sfpu_for_each_face_
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
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_out);
        if constexpr (Op::walks_faces)
        {
            _llk_math_eltwise_ternary_sfpu_run_<TENSOR_SHAPE>(Op::calculate, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
        }
        else
        {
            static_assert(_llk_math_eltwise_sfpu_is_full_tile_(TENSOR_SHAPE), "An op that walks Dest itself processes the full tile");
            _llk_math_eltwise_ternary_sfpu_run_once_(Op::calculate, dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
        }
    }

    /**
     * @brief Bridge for compute APIs that still take a VectorMode; new code uses run<TENSOR_SHAPE>().
     *
     * RC, R and C run a 32x32, 16x32 and 32x16 tile; None and RC_custom run a single 16x16 face.
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
            run<DEFAULT_TENSOR_SHAPE>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else if (vector_mode == VectorMode::R)
        {
            run<tensor_shape_from_tile_dims(MAX_FACE_R_DIM, MAX_TILE_C_DIM)>(
                dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        }
        else if (vector_mode == VectorMode::C)
        {
            run<tensor_shape_from_tile_dims(MAX_TILE_R_DIM, MAX_FACE_C_DIM)>(
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
     * @tparam TENSOR_SHAPE: Tile to process; see @ref _llk_math_eltwise_sfpu_for_each_face_
     * @param sfpu_func: SFPU functor that processes one face per call
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to sfpu_func
     */
    template <TensorShape TENSOR_SHAPE, typename Callable, typename... Args>
    static inline __attribute__((always_inline)) void run(
        Callable&& sfpu_func,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_out);
        _llk_math_eltwise_ternary_sfpu_run_<TENSOR_SHAPE>(
            std::forward<Callable>(sfpu_func), dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }

    /**
     * @brief Run a functor that walks Dest itself, calling it exactly once.
     *
     * @param sfpu_func: SFPU functor
     * @param dst_index_in0: Dest tile index of the first input
     * @param dst_index_in1: Dest tile index of the second input
     * @param dst_index_in2: Dest tile index of the third input
     * @param dst_index_out: Dest tile index that receives the result
     * @param args: Extra arguments passed to sfpu_func
     */
    template <typename Callable, typename... Args>
    static inline __attribute__((always_inline)) void run_once(
        Callable&& sfpu_func,
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        Args&&... args)
    {
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in0);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in1);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_in2);
        _llk_math_eltwise_sfpu_check_dst_index_(dst_index_out);
        _llk_math_eltwise_ternary_sfpu_run_once_(
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
