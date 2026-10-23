// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sanitizer/api.h"
#include "sfpu/common.h"
#include "sfpu/elementwise.h"
#include "tensor_shape_coverage_math.h"

namespace llk::sfpu
{

/// Base of every op @ref UnaryDispatch runs.
struct SfpuUnary
{
};

template <typename Op>
struct UnaryDispatch;

/**
 * @brief Base of a unary op's Init entry point.
 *
 * Its constructor is reachable only from @ref UnaryDispatch, so an Init derived from it (and left an
 * aggregate) can only be obtained through @ref UnaryDispatch::init.
 *
 * @tparam Op: Unary SFPU op the Init belongs to
 */
template <typename Op>
class UnaryInit
{
    friend UnaryDispatch<Op>;

protected:
    UnaryInit()
    {
    }
};

/**
 * @brief Base of a unary op's Execute entry point: carries the Dest tiles the dispatcher checked.
 *
 * Its constructor is reachable only from @ref UnaryDispatch, so an Execute derived from it (and left
 * an aggregate) can only be obtained through @ref UnaryDispatch::execute.
 *
 * @tparam Op: Unary SFPU op the Execute belongs to
 */
template <typename Op>
class UnaryExecute
{
    friend UnaryDispatch<Op>;

protected:
    UnaryExecute(const std::uint32_t output, const std::uint32_t input) : output(output), input(input)
    {
    }

    /// Dest tile the op writes, bounds-checked by @ref UnaryDispatch::execute.
    const std::uint32_t output;

    /// Dest tile the op reads, bounds-checked by @ref UnaryDispatch::execute. Equal to output in place.
    const std::uint32_t input;
};

/**
 * @brief Static dispatcher for unary SFPU ops: binds the parameters shared by every unary op, checks
 *        them, and hands back the op's entry point.
 *
 * The op contract:
 * @code
 * struct Op : SfpuUnary
 * {
 *     static constexpr PaddingAgnostic padding_agnostic();
 *
 *     struct Init : UnaryInit<Op>
 *     {
 *         template <...> void run(...) const;
 *     };
 *
 *     template <const TensorShape& SHAPE, PaddingAgnostic PADDING>
 *     struct Execute : UnaryExecute<Op>
 *     {
 *         template <...> void run(...) const;   // e.g. unary<PADDING>(DestTensor<V, SHAPE>(this->output), DestTensor<V, SHAPE>(this->input), ...)
 *     };
 * };
 * @endcode
 *
 * @ref UnaryInit and @ref UnaryExecute are constructible only by the dispatcher, so it is the only
 * way to obtain Init / Execute and its checks cannot be bypassed.
 *
 * @tparam Op: Unary SFPU op, derived from @ref SfpuUnary
 */
template <typename Op>
struct UnaryDispatch
{
    static_assert(std::is_base_of_v<SfpuUnary, Op>, "UnaryDispatch runs ops derived from llk::sfpu::SfpuUnary");

    /**
     * @brief Program the SFPU state shared by every unary op and return the op's init entry point.
     */
    static inline __attribute__((always_inline)) auto init()
    {
        SAN_HOOK(unsupported());
        _llk_math_eltwise_unary_sfpu_init_once_();
        math::reset_counters(p_setrwc::SET_ABD_F);

        static_assert(std::is_base_of_v<UnaryInit<Op>, typename Op::Init>, "Op::Init must derive from UnaryInit<Op>");
        return typename Op::Init {};
    }

    /**
     * @brief Check the shared parameters of one op call and return the op's execute entry point.
     *
     * @tparam SYNC: Dest sync mode the tile index is checked against, values = <SyncHalf/SyncFull>
     * @tparam SHAPE: Region of the Dest tile the op covers
     * @tparam PADDING: Padding contract of this call, passed on to the op's Execute. Defaults to
     *         Op::padding_agnostic(); a caller overrides it either way, e.g. No when the result feeds a
     *         padding-sensitive consumer, Yes when it has masked the padding of a sensitive op.
     * @param output: Dest tile the op writes
     * @param input: Dest tile the op reads; pass output to operate in place
     */
    template <DstSync Sync, const TensorShape& Shape, PaddingAgnostic Agnostic = Op::padding_agnostic()>
    static inline __attribute__((always_inline)) auto execute(const std::uint32_t output, const std::uint32_t input)
    {
        static_assert(
            std::is_same_v<decltype(Op::padding_agnostic()), PaddingAgnostic>, "the op must declare static constexpr PaddingAgnostic padding_agnostic()");

        LLK_VALIDATE_TENSOR_SHAPE_MATH("llk::sfpu::UnaryDispatch::execute", Shape);
        _llk_math_eltwise_sfpu_assert_dst_index_<Sync>(output, "output exceeds max dest tiles");
        _llk_math_eltwise_sfpu_assert_dst_index_<Sync>(input, "input exceeds max dest tiles");
        SAN_HOOK(unsupported());

        static_assert(std::is_base_of_v<UnaryExecute<Op>, typename Op::template Execute<Shape, Agnostic>>, "Op::Execute must derive from UnaryExecute<Op>");
        return typename Op::template Execute<Shape, Agnostic> {{output, input}};
    }
};

} // namespace llk::sfpu
