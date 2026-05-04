// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief Initializes unpacker to unpack operand 0 (buf_desc_id_0) into SrcB
 * and unpacks operand 1 (buf_desc_id_1) into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
 * In order to get output of rowmajor matrix multiplication input 0 * Input 1, need to initialize
 * SrcA and SrcB to be input 1 & input 0 respectively.
 * The following matrix multiply has the following dimensions:
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 32
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
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
 * @brief Initializes unpacker to unpack operand 0 (buf_desc_id_0) into SrcB
 * and unpacks operand 1 (buf_desc_id_1) into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
 * In order to get output of rowmajor matrix multiplication input 0 * Input 1, need to initialize
 * SrcA and SrcB to be input 1 & input 0 respectively.
 * The following matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, currently only supported for SrcA,
 * but can support other unpackers
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
template <bool TRANSPOSE_EN>
inline void _llk_unpack_matmul_init_(std::uint32_t buf_desc_id_0, std::uint32_t buf_desc_id_1, std::uint8_t ct_dim, std::uint8_t rt_dim, std::uint32_t kt_dim)
{
    static_assert((TRANSPOSE_EN == false), "TODO: Transpose srcA not available yet");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);

    _llk_unpack_matmul_mop_config_(buf_desc_id_0, buf_desc_id_1, ct_dim, rt_dim, kt_dim);
}

/**
 * @brief Performs unpack operation for matrix multiply such that:
 * Input 0 -> unpack to SrcB
 * Input 1 -> unpack to SrcA
 * Performs unpack for rt & ct dims of input0 & input 1 respectively
 * The following matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER1 -> SRCB
 * start_l1_tile_idx_1 -> UNPACKER0 -> SRCA
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
