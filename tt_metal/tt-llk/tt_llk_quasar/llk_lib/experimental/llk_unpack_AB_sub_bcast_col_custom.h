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
// SrcB is unpacked ONCE for the whole block row as a single whole-tile UNPACR1 (one dvalid, faces laid
// out linearly in the SrcB register) and then held while ct_dim SrcA tiles stream past it. The COL
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
 * @param tensor_shape: Operand tile shape (4 faces for full 32x32 tiles).
 * @note On the math thread, pair with @ref _llk_math_eltwise_binary_init_custom_ (T1); on pack, with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_AB_sub_bcast_col_custom_ is the matching execute call on this thread.
 */
inline void _llk_unpack_AB_sub_bcast_col_init_custom_(const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
}

/**
 * @brief SDPA blocked bcast-col unpack: one held SrcB tile + ct_dim SrcA tiles (Quasar).
 *
 * SrcB is unpacked once as a whole tile with a single dvalid and is not advanced or re-issued, so it
 * stays resident for the whole block; the math thread reads its faces by counter arithmetic and
 * releases it only after the last tile. Then ct_dim SrcA tiles are unpacked, each setting dvalid and
 * advancing the SrcA tile pointer. The SrcA loop self-paces on the SrcA bank-valid interlock
 * (UNPACR0 stalls when both SrcA banks are full), so it stays at most one tile ahead of the math
 * thread.
 *
 * @param buf_desc_id_0: Buffer-descriptor id feeding UNPACKER0 -> SrcA (operandA id).
 * @param buf_desc_id_1: Buffer-descriptor id feeding UNPACKER1 -> SrcB (operandB id).
 * @param start_l1_tile_idx_0: SrcA base tile index in L1 (first of the ct_dim tiles).
 * @param start_l1_tile_idx_1: SrcB tile index in L1 (the single held tile).
 * @param ct_dim: Number of SrcA column tiles unpacked (SrcB is unpacked once regardless).
 * @param tensor_shape: Operand tile shape (face count comes from the buffer descriptor; kept for API parity).
 * @note Call @ref _llk_unpack_AB_sub_bcast_col_init_custom_ first.
 */
inline void _llk_unpack_AB_sub_bcast_col_custom_(
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const std::uint32_t ct_dim                                = 1,
    [[maybe_unused]] const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // Position SrcA/SrcB read counters at their L1 tile bases; reset the SrcA/SrcB dest tile counters.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // SrcB: unpack the whole tile ONCE, set dvalid, hold (do not advance the SrcB tile pointer).
    TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx Inc*/, 0 /*Src Tile Idx Inc: pinned*/, buf_desc_id_1, 1 /*Set Dvalid*/);

    // SrcA: ct_dim tiles, each sets dvalid and advances the SrcA tile pointer by one.
    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        TT_UNPACR0_TILE_INC(0 /*Dst Tile Idx Inc*/, 1 /*Src Tile Idx Inc*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
}
