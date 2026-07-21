// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief Builds the MOP that unpacks operand 0 into SrcB and operand 1 into SrcA for matrix multiply.
 *
 * The matrix multiply FPU operation computes SrcB * SrcA. To obtain the row-major result
 * Input0 * Input1, SrcA and SrcB are loaded from Input1 and Input0 respectively. This unpacker only
 * sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]; kt_dim is assumed to be iterated over outside this
 * call. Constraints: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 *
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 32
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 */
inline void _llk_unpack_matmul_mop_config_(
    std::uint32_t buf_desc_id_0, std::uint32_t buf_desc_id_1, std::uint8_t ct_dim, std::uint8_t rt_dim, std::uint32_t kt_dim)
{
    const bool reuse_a                     = ct_dim >= rt_dim;
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = reuse_a ? ct_dim : rt_dim;
    std::uint32_t unpack_instrn;
    // static uint inc_l1_instrn;
    std::uint32_t unpack_reuse_instrn;

    if (reuse_a)
    {
        unpack_instrn = TT_OP_UNPACR0_TILE_INC(0, 1, buf_desc_id_1, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 1);
        unpack_reuse_instrn = TT_OP_UNPACR1_TILE_INC(0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_instrn = TT_OP_UNPACR1_TILE_INC(0, kt_dim, buf_desc_id_0, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, KT_DIM);
        unpack_reuse_instrn = TT_OP_UNPACR0_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn /*, inc_l1_instrn*/);
    temp.set_start_op(unpack_reuse_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack operand 0 into SrcB and operand 1 into SrcA for matrix multiply.
 *
 * The matrix multiply FPU operation computes SrcB * SrcA. To obtain the row-major result
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim], SrcA and SrcB are loaded
 * from Input1 and Input0 respectively. This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim];
 * kt_dim is assumed to be iterated over outside this call. Constraints: ct_dim * rt_dim <= 8 tiles in
 * a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 *
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, currently only supported for SrcA but can support other unpackers, values = <true/false>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 * @note On the math thread, pair with @ref _llk_math_matmul_init_ (T1); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_matmul_ is the matching execute call on this thread.
 */
template <bool TRANSPOSE_EN>
inline void _llk_unpack_matmul_init_(std::uint32_t buf_desc_id_0, std::uint32_t buf_desc_id_1, std::uint8_t ct_dim, std::uint8_t rt_dim, std::uint32_t kt_dim)
{
    static_assert((TRANSPOSE_EN == false), "TODO: Transpose srcA not available yet");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    _llk_unpack_matmul_mop_config_(buf_desc_id_0, buf_desc_id_1, ct_dim, rt_dim, kt_dim);
}

/**
 * @brief Unpacks operands for matrix multiply: Input0 -> SrcB, Input1 -> SrcA.
 *
 * Unpacks for the rt and ct dims of input0 and input1 respectively, producing
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]. This unpacker only sets
 * up Input0 [rt_dim, 1] x Input1 [1, ct_dim]; kt_dim is assumed to be iterated over outside this call.
 * Constraints: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 *
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER1 -> SRCB, start_l1_tile_idx_1 -> UNPACKER0 -> SRCA.
 * @note Call @ref _llk_unpack_matmul_init_ with matching template args before this function.
 */
inline void _llk_unpack_matmul_(
    std::uint8_t ct_dim, std::uint8_t rt_dim, std::uint32_t kt_dim, const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    // Reset Dest counters for Unpacker to 0
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t tile_idx_0 = start_l1_tile_idx_0 + (reuse_a ? (t * kt_dim) : 0);
        std::uint32_t tile_idx_1 = start_l1_tile_idx_1 + (reuse_a ? (0) : (t));

        // Set Source counter to L1 base + offset
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, tile_idx_0);
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, tile_idx_1);

        // Runs MOP
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
