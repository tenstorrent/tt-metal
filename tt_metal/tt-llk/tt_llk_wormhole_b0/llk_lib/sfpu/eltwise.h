// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include "sfpu/common.h"
#include "sfpu/traverse.h"

namespace llk::sfpu
{

namespace detail
{

/**
 * @brief output = functor(inputs...), element by element, over the region the operands share.
 *
 * Operands may alias; an input may be the output.
 *
 * @tparam Agnostic: Whether the call may compute on lanes outside the region. See @ref PaddingAgnostic.
 * @tparam Output: @ref DestTensor written
 * @tparam Functor: Callable Inputs::vType... -> Output::vType, deduced from @p functor
 * @tparam Inputs: @ref DestTensor read, one per functor argument
 * @note A complete SFPU call: needs nothing around it other than the op's own init.
 */
template <PaddingAgnostic Agnostic, typename Output, typename Functor, typename... Inputs>
inline __attribute__((always_inline)) void eltwise(const Output output, Functor&& functor, const Inputs... inputs)
{
    static_assert(is_dest_tensor_v<Output> && (is_dest_tensor_v<Inputs> && ...), "Operand must be a DestTensor");
    static_assert(Output::writable(), "Output must be writable to Dest");
    static_assert((Inputs::readable() && ...), "Input must be readable from Dest");
    static_assert(((Inputs::shape == Output::shape) && ...), "All operands must have the same shape");
    static_assert(is_walkable_shape(Output::shape), "Shape must be a Tile");
    static_assert(Agnostic == PaddingAgnostic::Yes || !walk_touches_padding(Output::shape), "a padding-sensitive call cannot touch padding");
    static_assert(
        std::is_invocable_r_v<typename Output::vType, Functor&, typename Inputs::vType...>,
        "functor must be callable on the inputs' vector types and return something convertible to the output's");

    walk_vectors<Output::shape>(
        [&]() __attribute__((always_inline))
        { sfpi::dst_reg[output.index * DEST_TILE_STRIDE] = functor(static_cast<typename Inputs::vType>(sfpi::dst_reg[inputs.index * DEST_TILE_STRIDE])...); });
}

} // namespace detail

/// output = functor(input). Pass the same tensor twice to operate in place.
template <PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Output, typename Input, typename Functor>
inline __attribute__((always_inline)) void eltwise_unary(const Output output, const Input input, Functor&& functor)
{
    detail::eltwise<Agnostic>(output, std::forward<Functor>(functor), input);
}

/// output = functor(a, b). Any operand may alias another.
template <PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Output, typename Left, typename Right, typename Functor>
inline __attribute__((always_inline)) void eltwise_binary(const Output output, const Left left, const Right right, Functor&& functor)
{
    detail::eltwise<Agnostic>(output, std::forward<Functor>(functor), left, right);
}

/// output = functor(a, b, c). Any operand may alias another.
template <PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Output, typename First, typename Second, typename Third, typename Functor>
inline __attribute__((always_inline)) void eltwise_ternary(const Output output, const First first, const Second second, const Third third, Functor&& functor)
{
    detail::eltwise<Agnostic>(output, std::forward<Functor>(functor), first, second, third);
}

} // namespace llk::sfpu
