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

/**
 * @brief Call @p body @p Count times with @p Unroll iteratations being unrolled into a single iteration.
 *
 * This exists because `#pragma GCC unroll` doesn't accept expressions, but does accept templates parameters.
 *
 * @tparam Count: Number of calls
 * @tparam Unroll: Unrolling factor for the loop
 */
template <std::size_t Count, std::size_t Unroll, typename Body>
[[gnu::always_inline]] inline void repeat(Body&& body)
{
#pragma GCC unroll Unroll
    for (std::size_t i = 0; i < Count; i++)
    {
        body();
    }
}

} // namespace detail

/**
 * Step the Dest RWC by the given number of vectors
 *
 * Model:
 * @code
 * Rows = Vectors << 1;
 * RWC.Dest += Rows;
 * @endcode
 *
 * @tparam Vectors: Number of vectors to step
 */
template <std::size_t Vectors>
inline void step_vectors()
{
    constexpr std::size_t rows = Vectors << 1; // Physical row

    // INCRWC Increment field has max value of 15
    constexpr std::size_t full      = rows / 15;
    constexpr std::size_t remainder = rows % 15;

    detail::repeat<full, full>([] { TTI_INCRWC(0, 15, 0, 0); });

    if constexpr (remainder > 0)
    {
        TTI_INCRWC(0, remainder, 0, 0);
    }
}

/**
 * Step the Dest RWC by the given number of faces RELATIVE TO BASE
 *
 * Model:
 * @code
 * Rows = Faces * 8 << 1;
 * RWC.DestCr += Rows;
 * RWC.Dest = DestCr;
 * @endcode
 *
 * @tparam Faces: Number of faces to step
 */
template <std::size_t Faces>
inline void step_faces()
{
    constexpr std::size_t rows = (Faces * 8) << 1; // Physical row

    // SETRWC Dest field has max value of 15
    constexpr std::size_t full      = rows / 15;
    constexpr std::size_t remainder = rows % 15;

    detail::repeat<full, full>([] { TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 15, 0, 0, p_setrwc::SET_D); });

    if constexpr (remainder > 0)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, remainder, 0, 0, p_setrwc::SET_D);
    }
}

/**
 * @brief Checks if walking @p shape stays inside it, so any op can walk it, padding-sensitive or not.
 *
 * @param shape: TensorShape that is being walked
 * @return True if the walk does not touch padding. False if the face height is not a multiple of DEST_VECTOR_ROWS
 *         or the face is narrower than MAX_FACE_C_DIM, so only a padding-agnostic op can walk @p shape.
 */
constexpr bool is_walk_agnostic(const TensorShape& shape)
{
    // Rows are covered in whole bands of DEST_VECTOR_ROWS, and every band covers all MAX_FACE_C_DIM columns.
    const bool full_row_groups = shape.face_r_dim % DEST_VECTOR_ROWS == 0;
    const bool full_rows       = shape.face_c_dim == MAX_FACE_C_DIM;

    return full_row_groups && full_rows;
}

/**
 * @brief Checks if the shape is standard tile.
 *
 * Criteria:
 * - No dimension is zero
 * - Faces are at most MAX_FACE_R_DIM x MAX_FACE_C_DIM
 * - The grid is at most MAX_NUM_FACES_R_DIM x MAX_NUM_FACES_C_DIM faces
 *
 * @param shape: TensorShape that is being walked
 * @return True if all criteria are met, false otherwise
 */
constexpr bool is_shape_walkable(const TensorShape& shape)
{
    constexpr auto in_range = [](const std::uint32_t value, const std::uint32_t low, const std::uint32_t high) { return low <= value && value <= high; };

    const bool face_fits = in_range(shape.face_r_dim, 1, MAX_FACE_R_DIM) && in_range(shape.face_c_dim, 1, MAX_FACE_C_DIM);
    const bool grid_fits = in_range(shape.num_faces_r_dim, 1, MAX_NUM_FACES_R_DIM) && in_range(shape.num_faces_c_dim, 1, MAX_NUM_FACES_C_DIM);

    return face_fits && grid_fits;
}

/// @return number of faces in a TensorShape
constexpr std::size_t face_count(const TensorShape& shape)
{
    return shape.num_faces_r_dim * shape.num_faces_c_dim;
}

/// @return number of vectors in a TensorShape
constexpr std::size_t vector_count(const TensorShape& Shape)
{
    // A vector is a noncontiguous DEST_VECTOR_ROWS x DEST_VECTOR_COLS block of the face:
    // - it spans DEST_VECTOR_ROWS rows, taking every DEST_VECTOR_COL_STEP-th column;
    // - the vectors that share a band of DEST_VECTOR_ROWS rows are interleaved column by column and consecutive
    //   in Dest (SFP_DESTREG_STRIDE apart). There are MAX_FACE_C_DIM / DEST_VECTOR_COLS of them per band; with
    //   two, Vec[2i] holds the band's even columns and Vec[2i + 1] its odd ones;
    // - a face narrower than MAX_FACE_C_DIM keeps its padding columns inside every vector of the band, so a face
    //   is always covered in whole bands and only face_r_dim shortens it.
    const std::size_t vector_count_columns = MAX_FACE_C_DIM / DEST_VECTOR_COLS;
    const std::size_t vector_count_rows    = (Shape.face_r_dim + DEST_VECTOR_ROWS - 1) / DEST_VECTOR_ROWS;
    return face_count(Shape) * vector_count_columns * vector_count_rows;
}

/**
 * @brief Call @p functor once per face of Shape, advancing Dest to the beginning of the next face after each call.
 * @note: The functor is allowed to leave the DEST pointer anywhere inside the face, as long as it doesn't touch DEST_CR
 *
 * @tparam Shape: Region of the Dest tile, at most 2x2 faces
 * @tparam UnrollFaces: Faces per loop iteration; 0 keeps the loop rolled
 * @param output_index: Dest tile the walk starts on
 * @param functor: Processes one face
 */
template <const TensorShape& Shape, std::size_t UnrollFaces = 0, typename FaceFunctor>
[[gnu::always_inline]] inline void walk_faces(const std::uint32_t output_index, FaceFunctor&& functor)
{
    static_assert(Shape.num_faces_r_dim >= 1 && Shape.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "the face walk supports 1 or 2 rows of faces");
    static_assert(Shape.num_faces_c_dim >= 1 && Shape.num_faces_c_dim <= MAX_NUM_FACES_C_DIM, "the face walk supports 1 or 2 columns of faces");

    // A 1-column shape skips the face to the right of each one it visits.
    constexpr std::size_t face_stride = MAX_NUM_FACES_C_DIM / Shape.num_faces_c_dim;

    CalculationScope scope(output_index);

    detail::repeat<face_count(Shape), UnrollFaces>(
        [&]
        {
            functor();
            step_faces<face_stride>();
        });
}

} // namespace llk::sfpu
