// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/common.h"
#include "tensor_shape.h"

namespace llk::sfpu
{

/**
 * @brief Whether a call may compute on Dest lanes outside the region it is given.
 */
enum class PaddingAgnostic
{
    Yes,
    No,
};

/// sfpi::dst_reg index distance between consecutive Dest tiles: a tile is 64 Dest rows.
constexpr std::uint32_t DEST_TILE_STRIDE = (MAX_NUM_FACES * MAX_FACE_R_DIM) / sfpi::SFP_DESTREG_STRIDE;

namespace detail
{

/// Advance the Dest address by Faces faces, expanded at compile time.
template <int Faces>
inline __attribute__((always_inline)) void step_faces()
{
    for (int face = 0; face < Faces; face++)
    {
        math::inc_dst_addr<8>();
        math::inc_dst_addr<8>();
    }
}

/// Whether the walk can cover @p shape: faces of at most 16x16, in a grid of at most 2x2.
constexpr bool is_walkable_shape(const TensorShape& shape)
{
    constexpr auto in_range = [](const std::uint32_t value, const std::uint32_t low, const std::uint32_t high) { return low <= value && value <= high; };

    const bool face_fits = in_range(shape.face_r_dim, 1, MAX_FACE_R_DIM) && in_range(shape.face_c_dim, 1, MAX_FACE_C_DIM);
    const bool grid_fits = in_range(shape.num_faces_r_dim, 1, MAX_NUM_FACES_R_DIM) && in_range(shape.num_faces_c_dim, 1, MAX_NUM_FACES_C_DIM);

    return face_fits && grid_fits;
}

/// Whether walking @p shape computes on lanes outside it.
constexpr bool walk_touches_padding(const TensorShape& shape)
{
    // Each SFPLOAD / SFPSTORE covers DEST_VECTOR_ROWS face rows by DEST_VECTOR_COLS columns of one
    // parity (even or odd), so the walk:
    // - covers rows in whole groups of DEST_VECTOR_ROWS: a face_r_dim that is not a multiple of it
    //   also walks the rows past it;
    // - covers every column of each row group: a face_c_dim below MAX_FACE_C_DIM also walks the
    //   columns past it.
    const bool partial_row_group = shape.face_r_dim % DEST_VECTOR_ROWS != 0;
    const bool partial_row       = shape.face_c_dim < MAX_FACE_C_DIM;

    return partial_row_group || partial_row;
}

} // namespace detail

/**
 * @brief Call @p face_fn once per face of the Dest region Shape selects.
 *
 * Faces are visited in row-major order starting at face 0 of the current Dest tile: a 2-column shape
 * steps one face after each call, a 1-column shape steps two (skipping the face to its right). Faces
 * after the last visited one are not stepped over; _llk_math_eltwise_sfpu_done_ resets the address.
 *
 * @tparam Shape: Region of the Dest tile to visit, at most 2x2 faces
 * @tparam FaceFn: Callable with no arguments, run with the Dest address at the start of a face
 * @param face_fn: Processes one face per call
 */
template <const TensorShape& Shape, typename FaceFunctor>
inline __attribute__((always_inline)) void for_each_face(FaceFunctor&& functor)
{
    static_assert(Shape.num_faces_r_dim >= 1 && Shape.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "the face walk supports 1 or 2 rows of faces");
    static_assert(Shape.num_faces_c_dim >= 1 && Shape.num_faces_c_dim <= MAX_NUM_FACES_C_DIM, "the face walk supports 1 or 2 columns of faces");

    // A 1-column shape also steps over the face to the right of each one it visits.
    constexpr int faces       = Shape.num_faces_r_dim * Shape.num_faces_c_dim;
    constexpr int face_stride = MAX_NUM_FACES_C_DIM / Shape.num_faces_c_dim;

#pragma GCC unroll 0
    for (int face = 0; face < faces; face++)
    {
        functor();
        detail::step_faces<face_stride>();
    }
}

/**
 * @brief Call @p functor once per sfpi vector of the Dest region Shape selects, stepping dst_reg after each.
 *
 * A face is walked in groups of DEST_VECTOR_ROWS rows, MAX_FACE_C_DIM / DEST_VECTOR_COLS vectors per
 * group (even columns, then odd columns).
 *
 * @tparam Shape: Region of the Dest tile to visit
 * @tparam VectorFunctor: Callable with no arguments, run with dst_reg at the current vector
 * @param functor: Called once per vector
 */
template <const TensorShape& Shape, typename VectorFunctor>
inline __attribute__((always_inline)) void for_each_vector(VectorFunctor&& functor)
{
    constexpr int vectors_per_face = (MAX_FACE_C_DIM / DEST_VECTOR_COLS) * ((Shape.face_r_dim + DEST_VECTOR_ROWS - 1) / DEST_VECTOR_ROWS);

    for_each_face<Shape>(
        [&]() __attribute__((always_inline))
        {
#pragma GCC unroll 8
            for (int vector = 0; vector < vectors_per_face; vector++)
            {
                functor();
                sfpi::dst_reg++;
            }
        });
}

} // namespace llk::sfpu
