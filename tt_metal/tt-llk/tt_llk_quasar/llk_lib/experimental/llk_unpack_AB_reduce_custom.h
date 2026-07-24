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
 * @brief Configures the unpack MOP for the block reduce_max_row operation.
 *
 * Unlike the Wormhole/Blackhole port (single legacy TT_OP_UNPACR + L1-address addressing), Quasar
 * uses per-unpacker, buffer-descriptor-based UNPACR ops. The scaler face (SrcB) is unpacked once via
 * the MOP start op, then @p block_ct_dim operand tiles are streamed into SrcA (one tile's faces per
 * outer-loop iteration). The math thread drains SrcA (CLR_A) between tiles, so the unpacker and math
 * pipeline over the whole block.
 *
 * @tparam block_ct_dim  Number of operand tiles in the width dimension processed as one block.
 * @param buf_desc_id_0  Buffer-descriptor id feeding UNPACKER0 -> SrcA (operand).
 * @param buf_desc_id_1  Buffer-descriptor id feeding UNPACKER1 -> SrcB (scaler).
 * @param tensor_shape   Operand tile shape (4 faces for 32x32, 2 faces for a 16x32 tiny tile).
 * @param respect_trigger  Accepted for signature parity with WH/BH; the split-MOP handshake is not
 *        implementable on Quasar (no FPU_SFPU / UNPACK_MATH_DONE semaphores), so it must be false.
 * @note This is NOT a substitute for @ref _llk_unpack_reduce_mop_config_. It is specialized for the
 *       SDPA/softmax block row-max path.
 */
template <std::uint32_t block_ct_dim>
inline void _llk_unpack_AB_reduce_block_max_row_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const bool respect_trigger = false)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    // respect_trigger (SDPA MOP-split handshake) relies on FPU_SFPU / UNPACK_MATH_DONE semaphores and
    // t6_semaphore_wait_on_zero, none of which exist on Quasar. Keep the parameter for API parity but
    // forbid enabling it here rather than silently mis-synchronizing.
    LLK_ASSERT(!respect_trigger, "respect_trigger is not supported on Quasar");

    const std::uint32_t MOP_OUTER_LOOP = block_ct_dim;
    const std::uint32_t MOP_INNER_LOOP = tensor_shape.total_num_faces();

    std::uint32_t unpack_srcA_face;
    std::uint32_t unpack_srcB_face;

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        unpack_srcA_face = TT_OP_UNPACR0_FACE_INC(0, 1 /*Src face Idx*/, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_srcB_face = TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_srcA_face = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_srcB_face = TT_OP_UNPACR1_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }

    // MAX pool over a partial (face_r_dim < FACE_R_DIM) face needs the unused SrcA rows seeded to -inf
    // so the pool ignores them; matches the native reduce unpacker path.
    const bool needs_srca_clear = (tensor_shape.face_r_dim < FACE_R_DIM);

    if (needs_srca_clear)
    {
        // MAX pool -> seed the cleared SrcA rows to -inf (matches native _llk_unpack_reduce_mop_config_).
        constexpr std::uint32_t clr_mode = p_unpacr::UNP_CLRSRC_NEGINF;
        const std::uint32_t unpack_zero_srcA =
            TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_zero_srcA, unpack_srcA_face);
        temp.set_start_op(unpack_srcB_face);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA_face);
        temp.set_start_op(unpack_srcB_face);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Enables/disables the UNPACKER0 hardware transpose used for row reduction.
 *
 * Row reduce transposes each SrcA face in the unpacker; the scaler (SrcB / UNPACKER1) is not
 * transposed. Split out from init because the metal layer sets this at init (no operands needed) but
 * programs the MOP later at execute (when the operand buffer-descriptor id is known).
 */
inline void _llk_unpack_AB_reduce_block_max_row_cfg_(const bool enable = true)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, enable ? 1 : 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
}

/**
 * @brief Initializes the unpacker for the block reduce_max_row operation.
 *
 * Enables the UNPACKER0 in-hardware transpose required for row reduction (UNPACKER1/scaler is not
 * transposed), then programs the block MOP. Used by the tt-llk test harness, which has the operand
 * buffer-descriptor ids at init time.
 *
 * @tparam block_ct_dim       Number of operand tiles processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode.
 * @tparam respect_trigger    Must be false on Quasar (see mop_config note).
 * @param buf_desc_id_0/1     Buffer-descriptor ids: 0 -> SrcA (operand), 1 -> SrcB (scaler).
 * @param tensor_shape        Operand tile shape.
 * @note On the math thread, pair with @ref _llk_math_reduce_block_max_row_init_; on pack, with
 *       @ref _llk_pack_reduce_mask_config_. @ref _llk_unpack_AB_reduce_block_max_row_ is the execute call.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    static_assert(!respect_trigger, "respect_trigger is not supported on Quasar");

    // Row reduce requires the SrcA face transpose in the unpacker; the scaler (SrcB) is not transposed.
    _llk_unpack_AB_reduce_block_max_row_cfg_(true);

    _llk_unpack_AB_reduce_block_max_row_mop_config_<block_ct_dim>(buf_desc_id_0, buf_desc_id_1, tensor_shape, respect_trigger);
}

/**
 * @brief Executes the block reduce_max_row unpack: streams @p block_ct_dim operand tiles into SrcA and
 *        the scaler face into SrcB.
 *
 * @param start_l1_tile_idx_0  Start L1 tile index for UNPACKER0 -> SrcA (operand).
 * @param start_l1_tile_idx_1  Start L1 tile index for UNPACKER1 -> SrcB (scaler).
 * @param tensor_shape         Operand tile shape.
 * @note Call @ref _llk_unpack_AB_reduce_block_max_row_init_ with matching template args first.
 */
template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_(
    const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape)
{
    static_assert(!respect_trigger, "respect_trigger is not supported on Quasar");

    const std::uint32_t l1_idx_A = (tensor_shape.total_num_faces() == NUM_FACES) ? start_l1_tile_idx_0 : start_l1_tile_idx_0 * tensor_shape.total_num_faces();
    const std::uint32_t l1_idx_B = (tensor_shape.total_num_faces() == NUM_FACES) ? start_l1_tile_idx_1 : start_l1_tile_idx_1 * tensor_shape.total_num_faces();

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_idx_A);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_idx_B);

    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Uninitializes the block reduce_max_row unpacker; disables the UNPACKER0 transpose.
 *
 * @tparam respect_trigger  Must be false on Quasar.
 */
template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_uninit_()
{
    static_assert(!respect_trigger, "respect_trigger is not supported on Quasar");
    _llk_unpack_AB_reduce_block_max_row_cfg_(false);
}
