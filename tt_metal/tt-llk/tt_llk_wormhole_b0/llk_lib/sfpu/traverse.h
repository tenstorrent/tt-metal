// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/common.h"
#include "tensor_shape.h"

namespace llk::sfpu
{

namespace detail
{

/// Advance Dest by Faces faces, relative to the counter base.
template <int Faces>
inline __attribute__((always_inline)) void step_faces()
{
    for (int face = 0; face < Faces; face++)
    {
        math::inc_dst_addr<8>();
        math::inc_dst_addr<8>();
    }
}

/// Whether @p shape fits the walk: faces of at most MAX_FACE_R_DIM x MAX_FACE_C_DIM, at most 2x2 of them.
constexpr bool is_walkable_shape(const TensorShape& shape)
{
    constexpr auto in_range = [](const std::uint32_t value, const std::uint32_t low, const std::uint32_t high) { return low <= value && value <= high; };

    const bool face_fits = in_range(shape.face_r_dim, 1, MAX_FACE_R_DIM) && in_range(shape.face_c_dim, 1, MAX_FACE_C_DIM);
    const bool grid_fits = in_range(shape.num_faces_r_dim, 1, MAX_NUM_FACES_R_DIM) && in_range(shape.num_faces_c_dim, 1, MAX_NUM_FACES_C_DIM);

    return face_fits && grid_fits;
}

/// Whether walking @p shape also touches lanes outside it.
constexpr bool walk_touches_padding(const TensorShape& shape)
{
    // Rows are covered in whole bands of DEST_VECTOR_ROWS, and every band covers all MAX_FACE_C_DIM columns.
    const bool partial_row_group = shape.face_r_dim % DEST_VECTOR_ROWS != 0;
    const bool partial_row       = shape.face_c_dim < MAX_FACE_C_DIM;

    return partial_row_group || partial_row;
}

} // namespace detail

/**
 * @brief Call @p functor once per face of Shape, advancing Dest to the next face after each call.
 * @note: In the current implementation, the functor may leave the DEST pointer anywhere inside the face. Plese don't rely on this behavior if you can, as we
 * might change it in the future to squeeze performance.
 *
 * @tparam Shape: Region of the Dest tile, at most 2x2 faces
 * @tparam Unroll: Faces per loop iteration; 0 keeps the loop rolled
 * @param output_index: Dest tile the walk starts on
 * @param functor: Processes one face
 */
template <const TensorShape& Shape, std::size_t Unroll = 0, typename FaceFunctor>
inline __attribute__((always_inline)) void walk_faces(const std::uint32_t output_index, FaceFunctor&& functor)
{
    static_assert(Shape.num_faces_r_dim >= 1 && Shape.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "the face walk supports 1 or 2 rows of faces");
    static_assert(Shape.num_faces_c_dim >= 1 && Shape.num_faces_c_dim <= MAX_NUM_FACES_C_DIM, "the face walk supports 1 or 2 columns of faces");

    // A 1-column shape skips the face to the right of each one it visits.
    constexpr std::size_t num_faces   = Shape.num_faces_r_dim * Shape.num_faces_c_dim;
    constexpr std::size_t face_stride = MAX_NUM_FACES_C_DIM / Shape.num_faces_c_dim;

    CalculationScope scope(output_index);

#pragma GCC unroll Unroll
    for (std::size_t face = 0; face < num_faces; face++)
    {
        functor();
        detail::step_faces<face_stride>();
    }
}

/**
 * @brief Call @p functor once per vector of one face of Shape, without advancing Dest.
 *
 * The functor must advance Dest by one vector itself. Use inside @ref walk_faces to cover several faces.
 *
 * @tparam Shape: Region of the Dest tile; only face_r_dim is used
 * @tparam Unroll: Vectors per loop iteration; 0 keeps the loop rolled
 * @param functor: Processes one vector
 */
template <const TensorShape& Shape, std::size_t UnrollVectors = 8, typename VectorFunctor>
inline __attribute__((always_inline)) void iterate_face_vectors(VectorFunctor&& functor)
{
    // A vector is a noncontiguous DEST_VECTOR_ROWS x DEST_VECTOR_COLS block of the face:
    // - it spans DEST_VECTOR_ROWS rows, taking every DEST_VECTOR_COL_STEP-th column;
    // - the vectors that share a band of DEST_VECTOR_ROWS rows are interleaved column by column and consecutive
    //   in Dest (SFP_DESTREG_STRIDE apart). There are MAX_FACE_C_DIM / DEST_VECTOR_COLS of them per band; with
    //   two, Vec[2i] holds the band's even columns and Vec[2i + 1] its odd ones;
    // - a face narrower than MAX_FACE_C_DIM keeps its padding columns inside every vector of the band, so a face
    //   is always covered in whole bands and only face_r_dim shortens it.
    constexpr std::size_t vector_count_columns = MAX_FACE_C_DIM / DEST_VECTOR_COLS;
    constexpr std::size_t vector_count_rows    = (Shape.face_r_dim + DEST_VECTOR_ROWS - 1) / DEST_VECTOR_ROWS;
    constexpr std::size_t vector_count         = vector_count_columns * vector_count_rows;

#pragma GCC unroll UnrollVectors
    for (std::size_t vector = 0; vector < vector_count; vector++)
    {
        functor();
    }
}

/**
 * @brief Call @p functor once per vector of Shape, advancing Dest after each call.
 *
 * @tparam Shape: Region of the Dest tile, at most 2x2 faces
 * @tparam UnrollVectors: Vectors per loop iteration; 0 keeps the loop rolled
 * @param functor: Processes one vector
 */
template <const TensorShape& Shape, std::size_t UnrollVectors = 8, typename VectorFunctor>
inline __attribute__((always_inline)) void walk_vectors(VectorFunctor&& functor)
{
    walk_faces<Shape>(
        0,
        [&]() __attribute__((always_inline))
        {
            iterate_face_vectors<Shape, UnrollVectors>(
                [&]() __attribute__((always_inline))
                {
                    functor();
                    sfpi::dst_reg++;
                });
        });
}

} // namespace llk::sfpu
