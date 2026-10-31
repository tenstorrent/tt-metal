// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"

namespace ckernel
{

/**
 * Where custom_mm puts its [{1, 2, 4, 8}, 32] output tiles in DEST. Each tile has two output faces of up to 8 rows,
 * or only output face 0 when in1 is a 2-face [32, 16] tile (see custom_mm_narrow); the face 1 slot then stays unused.
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

// Whether in1 is a 2-face [32, 16] tile ([16, 32] with transpose), giving [M, 16] output tiles, rather than a full
// [32, 32] tile. Both store their two faces in k order, so SrcA walks them the same way with or without transpose.
constexpr bool custom_mm_narrow(const std::uint32_t in1_num_faces)
{
    return in1_num_faces == CUSTOM_MM_NARROW_IN1_NUM_FACES;
}

// Whether the init and execute can take this in1 face count.
constexpr bool custom_mm_legal_in1_num_faces(const std::uint32_t in1_num_faces)
{
    return in1_num_faces == CUSTOM_MM_NARROW_IN1_NUM_FACES || in1_num_faces == MAX_NUM_FACES;
}

// Whether in1 has a supported shape: a full [32, 32] tile, or a narrow tile whose two faces run along k, which is
// [32, 16] without transpose and [16, 32] with it. A 2-face tile of the other orientation, or with short faces, also
// reports 2 faces, so the face count alone cannot tell them apart.
template <bool transpose>
constexpr bool custom_mm_legal_in1_shape(const TensorShape shape)
{
    const bool full   = shape.num_faces_r_dim == 2 && shape.num_faces_c_dim == 2;
    const bool narrow = transpose ? (shape.num_faces_r_dim == 1 && shape.num_faces_c_dim == 2) : (shape.num_faces_r_dim == 2 && shape.num_faces_c_dim == 1);
    return shape.face_r_dim == MAX_FACE_R_DIM && shape.face_c_dim == MAX_FACE_C_DIM && (full || narrow);
}

// Math replay length of one output tile: its MVMULs plus its finalization, which the execute replays when finalizing.
// A full tile records 4 MVMULs, 2 MOVD2Bs (4 when operandB_face_r_dim is 8), ZEROSRC and 2 ELWADDs: 9 or 11. A narrow
// tile records 2 MVMULs, 1 MOVD2B (2 when operandB_face_r_dim is 8), ZEROSRC and 1 ELWADD: 5 or 6. Keep in sync with
// the recording lambdas in llk_math_custom_mm.h.
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
