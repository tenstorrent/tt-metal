// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "experimental/llk_unpack_AB_reduce_custom.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

// Runtime-block_ct_dim variant of the block reduce_max_row unpacker. Same as the compile-time header
// but block_ct_dim is a runtime argument. See llk_unpack_AB_reduce_custom.h for details.

/**
 * @brief Runtime-block_ct_dim MOP config for the block reduce_max_row unpacker.
 */
inline void _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(
    const std::uint32_t block_ct_dim,
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const TensorShape& tensor_shape,
    const bool respect_trigger = false)
{
    // respect_trigger (SDPA MOP-split handshake) is not implementable on Quasar (no FPU_SFPU /
    // UNPACK_MATH_DONE semaphores). Keep the parameter for API parity but forbid enabling it.
    LLK_ASSERT(!respect_trigger, "respect_trigger is not supported on Quasar");

    // Per-tile delivery. A single MOP with outer=block_ct_dim + Src_Face_Idx_Inc does NOT advance the L1
    // source tile across the block -- it re-reads tile 0's faces every outer iteration (device-proven via
    // an SrcA dump: tiles 1..N-1 never appear in SrcA). So this MOP unpacks exactly ONE tile's faces into
    // SrcA; the execute fn loops block_ct_dim times, setting the L1 source-tile index per tile and
    // re-running this MOP -- that per-tile source-index set is what actually delivers tiles 1..N-1. The
    // scaler (SrcB) is constant for the whole block and is unpacked ONCE in the execute fn, not here.
    (void)block_ct_dim;  // block loop now lives in the execute fn, not the MOP outer loop
    (void)buf_desc_id_1; // scaler is unpacked directly in the execute fn
    const std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP = tensor_shape.total_num_faces();

    std::uint32_t unpack_srcA_face;
    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        // Dst_Face_Idx_Inc=0: every face is written to SrcA rows 0-15, and Set Dvalid flips the write
        // bank per face, so the two SrcA banks form a double-buffer that streams the faces to the math
        // thread (which pools them with ADDR_MOD_0 + CLR_SRCA_VLD, rotating the read bank per face).
        // Src_Face_Idx_Inc=1 walks THIS tile's 4 L1 faces.
        unpack_srcA_face = TT_OP_UNPACR0_FACE_INC(0 /*Dst face Idx inc*/, 1 /*Src face Idx inc*/, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_srcA_face = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    }

    const bool needs_srca_clear = (tensor_shape.face_r_dim < FACE_R_DIM);

    if (needs_srca_clear)
    {
        constexpr std::uint32_t clr_mode = p_unpacr::UNP_CLRSRC_NEGINF;
        const std::uint32_t unpack_zero_srcA =
            TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_zero_srcA, unpack_srcA_face);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA_face);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Runtime-block_ct_dim init for the block reduce_max_row unpacker.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_runtime_(
    const std::uint32_t block_ct_dim,
    const bool respect_trigger,
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!respect_trigger, "respect_trigger is not supported on Quasar");

    _llk_unpack_AB_reduce_block_max_row_cfg_(true);

    _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(block_ct_dim, buf_desc_id_0, buf_desc_id_1, tensor_shape, respect_trigger);
}

/**
 * @brief Runtime execute for the block reduce_max_row unpacker.
 */
inline void _llk_unpack_AB_reduce_block_max_row_runtime_(
    const std::uint32_t block_ct_dim,
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const std::uint32_t buf_desc_id_1,
    const TensorShape& tensor_shape,
    const bool respect_trigger    = false,
    const bool overlap_first_half = false)
{
    // respect_trigger / overlap_first_half rely on semaphores that do not exist on Quasar.
    LLK_ASSERT(!respect_trigger && !overlap_first_half, "respect_trigger/overlap_first_half are not supported on Quasar");

    const bool full_tiles = (tensor_shape.total_num_faces() == NUM_FACES);

    // Scaler (SrcB): constant for the whole block (MAX identity), so unpack it ONCE. The math pool reads
    // this one SrcB for every tile's GMPOOLs and clears it only at the very end.
    const std::uint32_t l1_idx_B = full_tiles ? start_l1_tile_idx_1 : start_l1_tile_idx_1 * tensor_shape.total_num_faces();
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_idx_B);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
    TT_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);

    // Operand (SrcA): unpack each block tile explicitly, advancing the L1 source-tile index per tile and
    // re-running the one-tile MOP. The MOP's Src_Face_Idx_Inc cannot cross the tile boundary on its own
    // (it re-reads tile 0), so the per-tile source-index set here is what actually delivers tiles 1..N-1.
    for (std::uint32_t tile = 0; tile < block_ct_dim; tile++)
    {
        const std::uint32_t tile_idx_A = start_l1_tile_idx_0 + tile;
        const std::uint32_t l1_idx_A   = full_tiles ? tile_idx_A : tile_idx_A * tensor_shape.total_num_faces();

        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_idx_A);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Runtime uninit for the block reduce_max_row unpacker.
 */
inline void _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(const bool respect_trigger = false, const bool overlap_first_half = false)
{
    LLK_ASSERT(!respect_trigger && !overlap_first_half, "respect_trigger/overlap_first_half are not supported on Quasar");
    _llk_unpack_AB_reduce_block_max_row_cfg_(false);
}
