// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Builds the block reduce_max_row unpack MOP.
 *
 * Per block tile: unpacks all operand faces into SrcA (transposed for reduce-row) and the scaler face
 * once into SrcB. buf_desc_id_0 feeds UNPACKER0 -> SRCA, buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 *
 * @tparam block_ct_dim: number of operand tiles in the block (MOP outer-loop count).
 * @tparam is_fp32_dest_acc_en: reserved for the 32-bit dest path; does not change the unpack stream.
 * @param buf_desc_id_0/1: buffer descriptor IDs for the operand (SrcA) and scaler (SrcB).
 * @param tensor_shape: operand tile shape (num faces, face dims).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_reduce_block_max_row_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const ckernel::TensorShape& tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");

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

    // Reduce-row with a full face row needs no SrcA padding; a partial face row (face_r_dim < 16) seeds
    // the missing rows with -Inf so the MAX pool ignores them.
    const bool needs_srca_clear = (tensor_shape.face_r_dim < FACE_R_DIM);

    if (needs_srca_clear)
    {
        const std::uint32_t unpack_zero_srcA =
            TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC_ZERO);

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
 * @brief Initializes the unpacker for block reduce_max_row.
 *
 * Enables SrcA transpose (reduce-row) and programs the block unpack MOP.
 *
 * @tparam block_ct_dim: number of operand tiles in the block.
 * @tparam is_fp32_dest_acc_en: enables the 32-bit-dest MOVB2D hi16/lo16 zero-flag config.
 * @param buf_desc_id_0/1: buffer descriptor IDs for the operand (SrcA) and scaler (SrcB).
 * @param tensor_shape: operand tile shape.
 * @note On the math thread, pair with @ref _llk_math_reduce_block_max_row_init_ (T1).
 * @note @ref _llk_unpack_reduce_block_max_row_ is the matching execute call on this thread.
 */
template <std::uint32_t block_ct_dim, std::uint32_t tile_count_x, std::uint32_t tile_count_y, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_reduce_block_max_row_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    if constexpr (is_fp32_dest_acc_en)
    {
        cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 1);
    }
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 1); // SrcA transpose ON (reduce-row)
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // scaler not transposed

    if constexpr (is_fp32_dest_acc_en)
    {
        _llk_unpack_reduce_block_max_row_mop_config_<1, true>(buf_desc_id_0, buf_desc_id_1, tensor_shape);
    }
    else if (tensor_shape.num_faces_r_dim == 1)
    {
        // Tiny tensors use a direct execute loop below because repeated Quasar unpack MOP
        // invocations can lose the Src-valid handoff at block boundaries.
        static_assert(block_ct_dim == 4, "16x32 block row max currently supports four tiles per block");
        static_assert(tile_count_x % block_ct_dim == 0, "Tiny block row max requires complete horizontal blocks");
        static_assert(tile_count_x * tile_count_y < 1024, "Tiny block row max tile count exceeds the Quasar limit");
    }
    else
    {
        _llk_unpack_reduce_block_max_row_mop_config_<block_ct_dim, false>(buf_desc_id_0, buf_desc_id_1, tensor_shape);
    }
}

/**
 * @brief Unpacks one block of operands (SrcA) and the scaler (SrcB) for block reduce_max_row.
 *
 * @param start_l1_tile_idx_0/1: start L1 tile indices; index 0 -> UNPACKER0 -> SRCA, index 1 -> UNPACKER1 -> SRCB.
 * @param tensor_shape: operand tile shape.
 * @note Call @ref _llk_unpack_reduce_block_max_row_init_ with matching template args before this function.
 */
template <std::uint32_t block_ct_dim, std::uint32_t tile_count_x, std::uint32_t tile_count_y, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_reduce_block_max_row_(
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const ckernel::TensorShape& tensor_shape)
{
    const auto unpack_tiles = [=](const std::uint32_t tile_offset)
    {
        const std::uint32_t tile_idx_A = start_l1_tile_idx_0 + tile_offset;
        const std::uint32_t tile_idx_B = start_l1_tile_idx_1 + tile_offset;
        const std::uint32_t l1_idx_A   = (tensor_shape.total_num_faces() == NUM_FACES) ? tile_idx_A : tile_idx_A * tensor_shape.total_num_faces();
        const std::uint32_t l1_idx_B   = (tensor_shape.total_num_faces() == NUM_FACES) ? tile_idx_B : tile_idx_B * tensor_shape.total_num_faces();

        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_idx_A);
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_idx_B);

        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    };

    if constexpr (is_fp32_dest_acc_en)
    {
        unpack_tiles(0);
        unpack_tiles(1);
    }
    else if (tensor_shape.num_faces_r_dim == 1)
    {
        const std::uint32_t l1_idx_B = start_l1_tile_idx_1 * tensor_shape.total_num_faces();

        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_idx_B);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

        // The fused kernel traverses horizontal blocks first, then tensor rows. Stream
        // SrcA in that same order so each math destination section consumes its own block.
        for (std::uint32_t block_x = 0; block_x < tile_count_x; block_x += block_ct_dim)
        {
            for (std::uint32_t block_y = 0; block_y < tile_count_y; ++block_y)
            {
                const std::uint32_t block_start_tile = start_l1_tile_idx_0 + block_y * tile_count_x + block_x;
                TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, block_start_tile * tensor_shape.total_num_faces());
                TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

                for (std::uint32_t tile = 0; tile < block_ct_dim; ++tile)
                {
                    // Native row-reduce releases SrcB after every logical tile, so refresh
                    // the scaler dvalid each time. The test scaler is constant, hence Src increment 0.
                    TTI_UNPACR1_TILE_INC(0, 0, buf_desc_id_1, 1);

                    for (std::uint32_t face = 0; face < 2; ++face)
                    {
                        TTI_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0, p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC_ZERO);
                        TTI_UNPACR0_TILE_INC(0, 1, buf_desc_id_0, 1);
                    }
                }
            }
        }
    }
    else
    {
        unpack_tiles(0);
    }
}

/**
 * @brief Restores unpacker state after block reduce_max_row (clears the reduce-row transpose config).
 */
inline void _llk_unpack_reduce_block_max_row_uninit_()
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
}
