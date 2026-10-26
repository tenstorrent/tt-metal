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

/**
 * todo(sstanisic): Document
 */
class EltwiseOptions
{
public:
    struct Struct
    {
        std::size_t face_unroll     = 0; // Unrolling factor for the outer, per face loop
        std::size_t vector_unroll   = 8; // Unrolling factor for the inner, per vector loop
    };

    constexpr EltwiseOptions& unroll_faces(const std::size_t faces)
    {
        options.face_unroll = faces;
        return *this;
    }

    constexpr EltwiseOptions& unroll_vectors(const std::size_t vectors)
    {
        options.vector_unroll = vectors;
        return *this;
    }

    constexpr Struct create() const
    {
        return options;
    }

private:
    Struct options {};
};

namespace detail
{
inline constexpr EltwiseOptions::Struct DefaultEltwiseOptions = EltwiseOptions {}.create();
} // namespace detail

/**
 * @brief Dest[output_index] = functor(Dest[input_index]...), element by element, over the region Shape selects.
 *
 * @tparam Shape: Region of each Dest tile to walk
 * @tparam Agnostic: Whether the call may compute on lanes outside the region. See @ref PaddingAgnostic.
 * @tparam Options: Loop unrolling, built with @ref EltwiseOptions
 * @tparam Functor: Computation on a single vector
 * @tparam Index: Dest tile index type of each input, one per functor argument
 * @note A complete SFPU call: needs nothing around it other than the op's own init.
 */
template <
    const TensorShape& Shape,
    PaddingAgnostic Agnostic,
    const EltwiseOptions::Struct& Options = detail::DefaultEltwiseOptions,
    typename Functor,
    typename... Index>
[[gnu::always_inline]] inline void eltwise(const std::uint32_t output_index, Functor&& functor, const Index... input_index)
{
    static_assert(((std::is_integral_v<Index> && !std::is_same_v<Index, bool>) && ...), "Input index must be a Dest tile index");
    static_assert(is_shape_walkable(Shape), "Shape must be a Tile");
    static_assert(Agnostic == PaddingAgnostic::Yes || is_walk_agnostic(Shape), "a padding-sensitive call cannot touch padding");

    constexpr std::size_t faces   = face_count(Shape);
    constexpr std::size_t vectors = vector_count(Shape) / faces;

    // A 1-column shape skips the face to the right of each one it visits.
    constexpr std::size_t face_stride  = MAX_NUM_FACES_C_DIM / Shape.num_faces_c_dim;
    constexpr std::size_t face_vectors = (MAX_FACE_R_DIM * MAX_FACE_C_DIM) / (DEST_VECTOR_ROWS * DEST_VECTOR_COLS); // Vectors in a full face
    constexpr std::size_t skip_vectors = face_stride * face_vectors - vectors;

    CalculationScope scope(output_index);

    // The scope points Dest at the output tile, so each input is addressed by its distance from it.
    const auto input_offset = [output_index](const std::uint32_t index)
    { return (static_cast<int>(index) - static_cast<int>(output_index)) * static_cast<int>(DEST_TILE_SIZE_VECTORS); };

    detail::repeat<faces, Options.face_unroll>(
        [&]
        {
            detail::repeat<vectors, Options.vector_unroll>(
                [&]
                {
                    // ADDR_MOD_2 under the SFPU addr mod base is ADDR_MOD_6: the store advances Dest by one vector.
                    auto output = sfpi::dst_reg[0].mode(ADDR_MOD_2);
                    output      = functor(sfpi::dst_reg[input_offset(input_index)]...);
                });

            // The vector loop left Dest `vectors` past the face start; step over the rest of it and any skipped face.
            step_vectors<skip_vectors>();
        });
}

/// Dest[output] = functor(Dest[input]). Pass the same index twice to operate in place.
template <const TensorShape& Shape, PaddingAgnostic Agnostic, const EltwiseOptions::Struct& Options = detail::DefaultEltwiseOptions, typename Functor>
[[gnu::always_inline]] inline void eltwise_unary(const std::uint32_t output, const std::uint32_t input, Functor&& functor)
{
    eltwise<Shape, Agnostic, Options>(output, std::forward<Functor>(functor), input);
}

/// Dest[output] = functor(Dest[left], Dest[right]). Any operand may alias another.
template <const TensorShape& Shape, PaddingAgnostic Agnostic, const EltwiseOptions::Struct& Options = detail::DefaultEltwiseOptions, typename Functor>
[[gnu::always_inline]] inline void eltwise_binary(const std::uint32_t output, const std::uint32_t left, const std::uint32_t right, Functor&& functor)
{
    eltwise<Shape, Agnostic, Options>(output, std::forward<Functor>(functor), left, right);
}

/// Dest[output] = functor(Dest[first], Dest[second], Dest[third]). Any operand may alias another.
template <const TensorShape& Shape, PaddingAgnostic Agnostic, const EltwiseOptions::Struct& Options = detail::DefaultEltwiseOptions, typename Functor>
[[gnu::always_inline]] inline void eltwise_ternary(
    const std::uint32_t output, const std::uint32_t first, const std::uint32_t second, const std::uint32_t third, Functor&& functor)
{
    eltwise<Shape, Agnostic, Options>(output, std::forward<Functor>(functor), first, second, third);
}

} // namespace llk::sfpu
