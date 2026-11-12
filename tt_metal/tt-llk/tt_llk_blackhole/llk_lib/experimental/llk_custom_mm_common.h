// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"

namespace ckernel
{

/**
 * Where custom_mm puts its [{1, 2, 4, 8}, 32] output tiles in DEST. Each tile has two output faces of up to 8 rows;
 * narrow in1 tiles (see custom_mm_narrow) fill only face 0.
 *
 *   Default          tiles 64 rows apart, output face 1 at row 16
 *   DefaultSplitAcc  as Default, accumulating the second k-face's partials at rows 8 and 24 for finalize to merge
 *   Dense            tiles 32 rows apart, output face 1 at row 16
 *   DenseSplitAcc    as Dense, with the split partials at rows 8 and 24
 *   Compact          tiles 16 rows apart, output face 1 at row 8, no room for split partials
 *
 * Init and execute must use the same layout, on the math and pack threads alike.
 *
 * The tile spacing sets how many output tiles, and so how wide a ct_dim, fit in DEST: 8 Default, 16 Dense or 32
 * Compact tiles in a 16-bit SyncHalf DEST (512 rows), twice that with SyncFull and half that with a 32-bit DEST.
 * See custom_mm_fits_dest.
 */
enum class CustomMmDestLayout : std::uint8_t
{
    Default,
    DefaultSplitAcc,
    Dense,
    DenseSplitAcc,
    Compact,
};

// DEST rows from one output tile to the next; dst_index counts tiles of this many rows.
constexpr std::uint32_t custom_mm_tile_rows(const CustomMmDestLayout layout)
{
    switch (layout)
    {
        case CustomMmDestLayout::Dense:
        case CustomMmDestLayout::DenseSplitAcc:
            return 32;
        case CustomMmDestLayout::Compact:
            return 16;
        default:
            return 64;
    }
}

constexpr bool custom_mm_split_acc(const CustomMmDestLayout layout)
{
    return layout == CustomMmDestLayout::DefaultSplitAcc || layout == CustomMmDestLayout::DenseSplitAcc;
}

// Faces in a narrow in1 tile: [32, 16], or [16, 32] with transpose. A full in1 tile has MAX_NUM_FACES.
constexpr std::uint32_t CUSTOM_MM_NARROW_IN1_NUM_FACES = MAX_NUM_FACES / 2;

// Whether in1 is a 2-face [32, 16] tile ([16, 32] with transpose); both keep their faces in k order.
constexpr bool custom_mm_narrow(const std::uint32_t in1_num_faces)
{
    return in1_num_faces == CUSTOM_MM_NARROW_IN1_NUM_FACES;
}

// Whether the init and execute can take this in1 face count.
constexpr bool custom_mm_legal_in1_num_faces(const std::uint32_t in1_num_faces)
{
    return in1_num_faces == CUSTOM_MM_NARROW_IN1_NUM_FACES || in1_num_faces == MAX_NUM_FACES;
}

// Whether in1 is a full [32, 32] tile, or a narrow one with both faces along k: [32, 16], or [16, 32] with transpose.
// Checks the shape, since a wrongly oriented or short-faced 2-face tile has the same face count.
template <bool transpose>
constexpr bool custom_mm_legal_in1_shape(const TensorShape shape)
{
    const bool full   = shape.num_faces_r_dim == 2 && shape.num_faces_c_dim == 2;
    const bool narrow = transpose ? (shape.num_faces_r_dim == 1 && shape.num_faces_c_dim == 2) : (shape.num_faces_r_dim == 2 && shape.num_faces_c_dim == 1);
    return shape.face_r_dim == MAX_FACE_R_DIM && shape.face_c_dim == MAX_FACE_C_DIM && (full || narrow);
}

// Math replay length of one output tile (MOVD2Bs doubled for 8-row in0 faces):
// full: 4 MVMUL + 2 MOVD2B + ZEROSRC + 2 ELWADD = 9 (11); narrow: 2 MVMUL + 1 MOVD2B + ZEROSRC + 1 ELWADD = 5 (6).
// Keep in sync with the replay recordings in llk_math_custom_mm.h.
constexpr std::uint32_t custom_mm_replay_buf_len(const std::uint32_t operandB_face_r_dim, const std::uint32_t in1_num_faces)
{
    if (custom_mm_narrow(in1_num_faces))
    {
        return operandB_face_r_dim == 8 ? 6 : 5;
    }
    return operandB_face_r_dim == 8 ? 11 : 9;
}

// Widest ct_dim the unpack MOP covers: the full unpack and 8 reuses, two more runs of 8 reuses, then 9 reuses ending
// in the final one. DEST capacity usually bounds it first.
constexpr std::uint32_t CUSTOM_MM_MAX_CT_DIM = 34;
// Widest kt_dim the SrcB address counters cover: 512 face increments split over the 8-bit Y and Z counters.
constexpr std::uint32_t CUSTOM_MM_MAX_KT_DIM = 256;

} // namespace ckernel
