// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/common.h"
#include "tensor_shape.h"

/**
 * Loops over the Dest region a TensorShape selects, in two families that differ in who moves the Dest
 * counter inside a face:
 *
 * - walk_*: the helper moves it. walk_vectors steps dst_reg after every call, so the functor computes on
 *   sfpi::dst_reg[0] and nothing else (the sfpi style).
 * - iterate_*: the helper only counts. The functor's own instructions move the counter, typically an
 *   SFPSTORE whose address mode increments Dest (the TTI style), and the vector count comes from the
 *   shape instead of an ITERATIONS constant.
 *
 * Both families step from face to face themselves: that step is relative to the counter base
 * (SETRWC with CR_D), so it lands on the next face whatever the functor did to the working counter.
 */
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

/**
 * @brief Number of sfpi vectors (one SFPLOAD / SFPSTORE each) that cover one face of Shape.
 *
 * A face is covered in groups of DEST_VECTOR_ROWS rows, MAX_FACE_C_DIM / DEST_VECTOR_COLS vectors per
 * group (even columns, then odd columns). A face_r_dim that is not a multiple of DEST_VECTOR_ROWS is
 * rounded up to whole groups.
 */
template <const TensorShape& Shape>
constexpr int VECTORS_PER_FACE = (MAX_FACE_C_DIM / DEST_VECTOR_COLS) * ((Shape.face_r_dim + DEST_VECTOR_ROWS - 1) / DEST_VECTOR_ROWS);

namespace detail
{

/// Advance the Dest address by Faces faces, expanded at compile time. Relative to the counter base.
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

/**
 * @brief Call @p functor once per face of the region Shape selects, stepping a face after each call.
 *
 * Faces are visited in row-major order starting at face 0 of the current Dest tile: a 2-column shape
 * steps one face after each call, a 1-column shape steps two (skipping the face to its right). The step
 * is relative to the counter base, so it is correct whether or not the functor moved the working counter.
 * Faces after the last visited one are not stepped over; _llk_math_eltwise_sfpu_done_ resets the address.
 *
 * @tparam Shape: Region of the Dest tile to visit, at most 2x2 faces
 * @tparam FaceFunctor: Callable with the arguments @p args, run with the Dest address at the start of a face
 */
template <const TensorShape& Shape, typename FaceFunctor, typename... Args>
inline __attribute__((always_inline)) void faces(FaceFunctor&& functor, Args... args)
{
    static_assert(Shape.num_faces_r_dim >= 1 && Shape.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "the face walk supports 1 or 2 rows of faces");
    static_assert(Shape.num_faces_c_dim >= 1 && Shape.num_faces_c_dim <= MAX_NUM_FACES_C_DIM, "the face walk supports 1 or 2 columns of faces");

    // A 1-column shape also steps over the face to the right of each one it visits.
    constexpr int num_faces   = Shape.num_faces_r_dim * Shape.num_faces_c_dim;
    constexpr int face_stride = MAX_NUM_FACES_C_DIM / Shape.num_faces_c_dim;

#pragma GCC unroll 0
    for (int face = 0; face < num_faces; face++)
    {
        functor(args...);
        step_faces<face_stride>();
    }
}

} // namespace detail

/**
 * @brief Call @p functor once per face of the region Shape selects; the helper moves between faces.
 *
 * What happens inside a face is the functor's: it is run with the Dest address at the start of the face
 * and may leave the working counter anywhere, the step to the next face does not depend on it.
 *
 * @tparam Shape: Region of the Dest tile to visit, at most 2x2 faces
 * @tparam FaceFunctor: Callable with no arguments
 * @param functor: Processes one face per call
 */
template <const TensorShape& Shape, typename FaceFunctor>
inline __attribute__((always_inline)) void walk_faces(FaceFunctor&& functor)
{
    detail::faces<Shape>(functor);
}

/**
 * @brief Call @p functor once per sfpi vector of the region Shape selects; the helper moves between vectors.
 *
 * dst_reg is stepped after every call, so the functor reads and writes sfpi::dst_reg[0] and moves nothing.
 *
 * @tparam Shape: Region of the Dest tile to visit
 * @tparam VectorFunctor: Callable with no arguments, run with dst_reg at the current vector
 * @param functor: Called once per vector
 */
template <const TensorShape& Shape, typename VectorFunctor>
inline __attribute__((always_inline)) void walk_vectors(VectorFunctor&& functor)
{
    detail::faces<Shape>(
        [&]() __attribute__((always_inline))
        {
#pragma GCC unroll 8
            for (int vector = 0; vector < VECTORS_PER_FACE<Shape>; vector++)
            {
                functor();
                sfpi::dst_reg++;
            }
        });
}

/**
 * @brief Call @p functor once per face of the region Shape selects, with the face's vector count; the
 *        functor owns the loop inside the face.
 *
 * For kernels that need a per-face prologue or epilogue around their own vector loop, for example an
 * SFPLOADMACRO sequence that loads constants first and drains with SFPNOPs after. The functor's
 * instructions move the working counter inside the face; the helper only steps from face to face.
 *
 * @tparam Shape: Region of the Dest tile to visit, at most 2x2 faces
 * @tparam FaceFunctor: Callable taking std::integral_constant<int, VECTORS_PER_FACE<Shape>>
 * @param functor: Processes one face per call, e.g. [](auto vectors) { for (int v = 0; v < vectors; v++) ... }
 */
template <const TensorShape& Shape, typename FaceFunctor>
inline __attribute__((always_inline)) void iterate_faces(FaceFunctor&& functor)
{
    detail::faces<Shape>(functor, std::integral_constant<int, VECTORS_PER_FACE<Shape>> {});
}

/**
 * @brief Call @p functor once per sfpi vector of the region Shape selects; the functor moves the counter.
 *
 * The helper issues no dst_reg step: the functor's instructions advance Dest by one vector per call, e.g.
 * an SFPSTORE through an address mode that increments Dest by DEST_VECTOR_ROWS rows. The step from face
 * to face is still the helper's.
 *
 * @tparam Shape: Region of the Dest tile to visit
 * @tparam VectorFunctor: Callable with no arguments, run with Dest at the current vector
 * @param functor: Called once per vector; must advance Dest by one vector
 */
template <const TensorShape& Shape, typename VectorFunctor>
inline __attribute__((always_inline)) void iterate_vectors(VectorFunctor&& functor)
{
    detail::faces<Shape>(
        [&]() __attribute__((always_inline))
        {
#pragma GCC unroll 8
            for (int vector = 0; vector < VECTORS_PER_FACE<Shape>; vector++)
            {
                functor();
            }
        });
}

} // namespace llk::sfpu
