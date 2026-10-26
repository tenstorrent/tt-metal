// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "sfpu/common.h"
#include "sfpu/traverse.h"

namespace llk::sfpu
{

namespace detail
{

/**
 * @brief Dest[output_index] = functor(Dest[input_index]...), element by element, over the region Shape selects.
 *
 * Operands may alias; an input may be the output.
 *
 * @tparam Shape: Region of each Dest tile to walk
 * @tparam Agnostic: Whether the call may compute on lanes outside the region. See @ref PaddingAgnostic.
 * @tparam Functor: Callable on SFPI vector types, deduced from @p functor. Its parameter types pick the format each
 * input is loaded from Dest in, so they must be concrete (vFloat, vInt, ...), not auto.
 * @tparam Index: Dest tile index type of each input, one per functor argument
 * @note A complete SFPU call: needs nothing around it other than the op's own init.
 */
template <const TensorShape& Shape, PaddingAgnostic Agnostic, typename Functor, typename... Index>
[[gnu::always_inline]] inline void eltwise(const std::uint32_t output_index, Functor&& functor, const Index... input_index)
{
    static_assert(((std::is_integral_v<Index> && !std::is_same_v<Index, bool>) && ...), "Input index must be a Dest tile index");
    static_assert(is_walkable_shape(Shape), "Shape must be a Tile");
    static_assert(Agnostic == PaddingAgnostic::Yes || !walk_touches_padding(Shape), "a padding-sensitive call cannot touch padding");

    walk_vectors<Shape>([&] [[gnu::always_inline]] ()
                        { sfpi::dst_reg[output_index * DEST_TILE_STRIDE] = functor(sfpi::dst_reg[input_index * DEST_TILE_STRIDE]...); });
}

} // namespace detail

/// Dest[output] = functor(Dest[input]). Pass the same index twice to operate in place.
template <const TensorShape& Shape, PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Functor>
[[gnu::always_inline]] inline void eltwise_unary(const std::uint32_t output, const std::uint32_t input, Functor&& functor)
{
    detail::eltwise<Shape, Agnostic>(output, std::forward<Functor>(functor), input);
}

/// Dest[output] = functor(Dest[left], Dest[right]). Any operand may alias another.
template <const TensorShape& Shape, PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Functor>
[[gnu::always_inline]] inline void eltwise_binary(const std::uint32_t output, const std::uint32_t left, const std::uint32_t right, Functor&& functor)
{
    detail::eltwise<Shape, Agnostic>(output, std::forward<Functor>(functor), left, right);
}

/// Dest[output] = functor(Dest[first], Dest[second], Dest[third]). Any operand may alias another.
template <const TensorShape& Shape, PaddingAgnostic Agnostic = PaddingAgnostic::No, typename Functor>
[[gnu::always_inline]] inline void eltwise_ternary(
    const std::uint32_t output, const std::uint32_t first, const std::uint32_t second, const std::uint32_t third, Functor&& functor)
{
    detail::eltwise<Shape, Agnostic>(output, std::forward<Functor>(functor), first, second, third);
}

} // namespace llk::sfpu
