// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Enable/disable the UNPACKER0 hardware transpose used for row reduction.
 *
 * Row reduce transposes each SrcA face in the unpacker; the scaler (SrcB / UNPACKER1) is not transposed.
 *
 * @param enable: true programs the transpose on (init); false clears it (uninit).
 */
inline void _llk_unpack_AB_reduce_block_max_row_cfg_(const bool enable = true)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::UNPACK0);
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, enable ? 1 : 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
}

/**
 * @brief Program the unpack MOP for block reduce_max_row: one outer iteration per tile, inner loop over
 *        the tile's faces; the last inner op advances the source tile so one run walks the whole block.
 *
 * @param block_ct_dim: Number of tiles in the block; becomes the MOP outer-loop count.
 * @param buf_desc_id_0: SrcA operand buffer descriptor id (baked into the UNPACR0 ops).
 * @param buf_desc_id_1: SrcB scaler buffer descriptor id; unused here -- the scaler is unpacked once in
 *                       the execute fn, not in this MOP.
 * @param tensor_shape: Operand tile shape; drives the face count and the partial-face -inf clear.
 * @param respect_trigger: SDPA MOP-split handshake; unsupported on Quasar, must stay false.
 * @note Called by @ref _llk_unpack_AB_reduce_block_max_row_init_runtime_ (and by the compile-time
 *       llk_api wrapper at execute); not a standalone entry point.
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

    // Single MOP walks the WHOLE block. Its outer loop = block_ct_dim (one tile per outer iteration); its
    // inner loop = the tile's faces.
    // The scaler (SrcB) is NOT in this MOP: the template's START_OP runs once PER OUTER ITERATION,
    // so putting the scaler there would re-copy the constant scaler block_ct_dim times.
    // It is unpacked exactly ONCE in the execute fn instead.
    (void)buf_desc_id_1; // scaler is unpacked once in the execute fn, not in this MOP
    const std::uint32_t MOP_OUTER_LOOP = block_ct_dim;
    const std::uint32_t MOP_INNER_LOOP = tensor_shape.total_num_faces();

    std::uint32_t unpack_srcA_face;      // faces 0..N-2 of a tile: advance the face index only
    std::uint32_t unpack_srcA_face_last; // last face of a tile: advance the face index AND the source tile

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        unpack_srcA_face      = TT_OP_UNPACR0_FACE_INC(0 /*Dst face*/, 1 /*Src face inc*/, 0, 0 /*Src tile off inc*/, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_srcA_face_last = TT_OP_UNPACR0_FACE_INC(0 /*Dst face*/, 1 /*Src face inc*/, 0, 1 /*Src tile off inc*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
    else
    {
        // Tiny path (not exercised here): TILE_INC already advances the source tile per op.
        unpack_srcA_face      = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_srcA_face_last = unpack_srcA_face;
    }

    // Partial-face tiles (face_r_dim < 16, e.g. 8x32 / 4x32) fill only the top face_r_dim SrcA rows.
    const bool needs_srca_clear = (tensor_shape.face_r_dim < FACE_R_DIM);

    if (needs_srca_clear)
    {
        // Seed the unfilled SrcA rows with -inf so GMPOOL's DIM_16X16 MAX over the full 16-row face
        // ignores them (max(x, -inf) = x). This clear op is prepended before each face unpack (the MOP's first inner-loop op).
        constexpr std::uint32_t clr_mode = p_unpacr::UNP_CLRSRC_NEGINF;
        const std::uint32_t unpack_zero_srcA =
            TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_zero_srcA, unpack_srcA_face);
        temp.set_last_inner_loop_instr(unpack_srcA_face_last);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA_face);
        temp.set_last_inner_loop_instr(unpack_srcA_face_last);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Configure the unpack thread for block reduce_max_row: enable the UNPACKER0 transpose and
 *        program the block MOP.
 *
 * @tparam is_fp32_dest_acc_en: 32-bit DEST accumulation, values = <false> (true unsupported on Quasar).
 * @param block_ct_dim: Number of tiles in the block, baked into the MOP.
 * @param respect_trigger: SDPA MOP-split handshake; unsupported on Quasar, must stay false.
 * @param buf_desc_id_0: SrcA operand buffer descriptor id.
 * @param buf_desc_id_1: SrcB scaler buffer descriptor id.
 * @param tensor_shape: Operand tile shape.
 * @note On the math thread, pair with @ref _llk_math_reduce_block_max_row_init_runtime_ (T1).
 * @note @ref _llk_unpack_AB_reduce_block_max_row_runtime_ runs the unpack; call
 *       @ref _llk_unpack_AB_reduce_block_max_row_uninit_runtime_ after to clear the transpose bit.
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
 * @brief Execute the block reduce_max_row unpack: unpack the constant scaler once into SrcB, then fire
 *        the block MOP to stream all block_ct_dim tiles' faces into the rotating SrcA banks.
 *
 * @param block_ct_dim: Number of tiles in the block (handled by the MOP outer loop; kept for symmetry).
 * @param start_l1_tile_idx_0: L1 tile index of the block's first SrcA tile.
 * @param start_l1_tile_idx_1: L1 tile index of the scaler (SrcB) tile.
 * @param buf_desc_id_1: SrcB scaler buffer descriptor id.
 * @param tensor_shape: Operand tile shape; selects full-tile vs partial-tile L1 index scaling.
 * @param respect_trigger: SDPA MOP-split handshake; unsupported on Quasar, must stay false.
 * @param overlap_first_half: SDPA MOP-split overlap; unsupported on Quasar, must stay false.
 * @note Call @ref _llk_unpack_AB_reduce_block_max_row_init_runtime_ first (programs the MOP) and
 *       @ref _llk_unpack_AB_reduce_block_max_row_uninit_runtime_ after. On T1, pair with
 *       @ref _llk_math_reduce_block_max_row_runtime_ which pools + transposes the delivered tiles.
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

    // The MOP (outer=block_ct_dim, last-inner op advances the source tile) walks the whole block in one
    // run. The scaler (SrcB) is constant for the whole block, so unpack it exactly ONCE here -- NOT inside
    // the MOP, whose per-outer-iteration START_OP would re-copy it block_ct_dim times. Then set the SrcA
    // block-start tile index and fire the MOP once.
    (void)block_ct_dim; // the MOP outer loop handles the block

    const bool full_tiles        = (tensor_shape.total_num_faces() == NUM_FACES);
    const std::uint32_t l1_idx_A = full_tiles ? start_l1_tile_idx_0 : start_l1_tile_idx_0 * tensor_shape.total_num_faces();
    const std::uint32_t l1_idx_B = full_tiles ? start_l1_tile_idx_1 : start_l1_tile_idx_1 * tensor_shape.total_num_faces();

    // Scaler (SrcB): unpacked ONCE for the whole block; the math reads this one SrcB for every tile's
    // GMPOOLs and clears it only at the very end.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_idx_B);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
    TT_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);

    // Operand (SrcA) block start; the MOP + its last-inner tile advance walk tiles 0..block_ct_dim-1.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_idx_A);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Uninit the unpack thread after block reduce_max_row: clear the UNPACKER0 transpose bit.
 *
 * @param respect_trigger: SDPA MOP-split handshake; unsupported on Quasar, must stay false.
 * @param overlap_first_half: SDPA MOP-split overlap; unsupported on Quasar, must stay false.
 * @note Pairs with @ref _llk_unpack_AB_reduce_block_max_row_init_runtime_.
 */
inline void _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(const bool respect_trigger = false, const bool overlap_first_half = false)
{
    LLK_ASSERT(!respect_trigger && !overlap_first_half, "respect_trigger/overlap_first_half are not supported on Quasar");
    _llk_unpack_AB_reduce_block_max_row_cfg_(false);
}
