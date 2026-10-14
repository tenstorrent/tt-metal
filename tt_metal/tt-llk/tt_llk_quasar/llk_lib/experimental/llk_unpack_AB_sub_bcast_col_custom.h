// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

// SDPA-specific custom blocked sub+bcast(col) unpack flow (Quasar).
//
// SrcB is unpacked ONCE for the whole block row by a single UNPACR1 (one dvalid, faces laid out
// linearly in the SrcB register) and then held while ct_dim SrcA tiles stream past it. The COL
// face pattern F0,F0,F2,F2 is NOT produced here: the math thread walks it with SrcB read-counter
// arithmetic (see @ref _llk_math_sub_bcast_cols_reuse_custom_). This mirrors the Blackhole flow.
//
// Issuing SrcB per face instead would force the math thread to advance the face pointer with p_elwise::CLR_SRCB_VLD,
// which also consumes the face, that is incompatible with holding SrcB across tiles.

/**
 * @brief Init the unpacker for the SDPA blocked bcast-col SUB path (Quasar).
 *
 * COL broadcast does not transpose, so clear the transpose RMW on both unpackers. The blocked unpack
 * is issued as a runtime instruction stream in @ref _llk_unpack_AB_sub_bcast_col_custom_ (no MOP),
 * so neither ct_dim nor the buffer descriptors are needed at init time.
 *
 * @param tensor_shape: Operand tile shape. 32x32 (2x2 faces) or 16x32 (1x2 faces).
 * @note On the math thread, pair with @ref _llk_math_eltwise_binary_init_custom_ (T1); on pack, with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_AB_sub_bcast_col_custom_ is the matching execute call on this thread.
 */
inline void _llk_unpack_AB_sub_bcast_col_init_custom_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // One predicate for all three threads; see @ref validate_tensor_shape_sub_bcast_col_custom_ for why
    // this path is stricter than validate_tensor_shape_tile_dependent_ops_.
    LLK_ASSERT(validate_tensor_shape_sub_bcast_col_custom_(tensor_shape), "custom sub bcast-col path supports 32x32 and 16x32 tiles only");

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
}

/**
 * @brief SDPA blocked bcast-col unpack: one held SrcB tile + ct_dim SrcA tiles (Quasar).
 *
 * SrcB is unpacked once with a single UNPACR and dvalid and is not advanced or re-issued, so it stays
 * resident for the whole block; the math thread reads its faces by counter arithmetic and releases it
 * only after the last tile. Then ct_dim SrcA tiles are unpacked, each raising exactly one dvalid on
 * its last face-UNPACR and advancing the SrcA L1 tile pointer. The SrcA loop self-paces on the SrcA
 * bank-valid interlock (UNPACR0 stalls when both SrcA banks are full), so it stays at most one tile
 * ahead of the math thread.
 *
 * A 16x32 tiny tile is registered as one HW tile per face (buffer-descriptor z_dim = 1, see
 * @ref ckernel::trisc::construct_buf_desc), so a SrcA tile takes one UNPACR per face and its L1 tile
 * indices count faces, not tiles. A full 32x32 tile is a single HW tile (z_dim = 4) unpacked by one
 * UNPACR. Either way exactly one dvalid is raised per tile, which is what the math thread's per-tile
 * CLR_A (and single CLR_B for the held SrcB) consumes. SrcB stays at one UNPACR in both cases: a full
 * tile arrives whole, and of a 16x32 tile the math walk reads face 0 only (COL broadcast reads column 0,
 * which lives in the face-row's first face), so unpacking face 1 would cost an instruction nothing reads.
 *
 * @param buf_desc_id_0: Buffer-descriptor id feeding UNPACKER0 -> SrcA (operandA id).
 * @param buf_desc_id_1: Buffer-descriptor id feeding UNPACKER1 -> SrcB (operandB id).
 * @param start_l1_tile_idx_0: SrcA base tile index in L1 (first of the ct_dim tiles).
 * @param start_l1_tile_idx_1: SrcB tile index in L1 (the single held tile).
 * @param ct_dim: Number of SrcA column tiles unpacked (SrcB is unpacked once regardless).
 * @param tensor_shape: Operand tile shape (drives the per-tile SrcA UNPACR count and L1 tile index scale).
 * @note Call @ref _llk_unpack_AB_sub_bcast_col_init_custom_ first.
 */
inline void _llk_unpack_AB_sub_bcast_col_custom_(
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const std::uint32_t ct_dim               = 1,
    const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool tiny_tile          = num_faces != ckernel::MAX_NUM_FACES;
    // One UNPACR per SrcA tile: a face for a tiny tile, the whole tile otherwise.
    const std::uint32_t unpacrs_per_tile = tiny_tile ? num_faces : 1;
    // Tiny-tile L1 indices are in faces (face_r_dim is 16 here, so one face per tile step).
    const std::uint32_t l1_tile_idx_scale = unpacrs_per_tile;
    // Advance the SrcA register tile counter per face so a tiny tile's faces land at consecutive
    // register rows; a full tile arrives in one UNPACR and needs no advance.
    const std::uint32_t dst_tile_idx_inc = tiny_tile ? 1 : 0;

    // Position SrcA/SrcB read counters at their L1 tile bases; reset the SrcA/SrcB dest tile counters.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * l1_tile_idx_scale);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1 * l1_tile_idx_scale);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // SrcB: ONE UNPACR, dvalid, then hold (never advance past this tile; the L1 read counter is re-set
    // on every call, so it stays pinned across block rows). For a full tile that is the whole tile,
    // faces 0-3; for a 16x32 tile it is face 0, the only SrcB face the math walk reads.
    TT_UNPACR1_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc: pinned*/, buf_desc_id_1, 1 /*SetDatValid*/);

    // SrcA: ct_dim tiles, each raising one dvalid on its last face and advancing the L1 tile pointer.
    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        for (std::uint32_t face = 0; face < unpacrs_per_tile; face++)
        {
            const bool last_face = (face + 1 == unpacrs_per_tile); // one dvalid per tile, on its last face
            TT_UNPACR0_TILE_INC(dst_tile_idx_inc, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id_0, last_face ? 1 : 0 /*SetDatValid*/);
        }

        if (tiny_tile)
        {
            // Rewind the SrcA register tile counter so the next tile's faces start at row 0 again.
            TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
        }
    }
}
