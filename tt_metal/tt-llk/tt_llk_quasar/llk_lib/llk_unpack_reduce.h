// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Builds the MOP for unpack reduce operations.
 *
 * Unpacks a full tile for SrcA and a single face for SrcB. buf_desc_id_0 feeds UNPACKER0 -> SRCA,
 * buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 * For REDUCE_ROW SUM/AVG the operands are swapped (see @ref reduce_swaps_operands).
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @param num_tiles: Number of tiles to unpack at a time for SrcA; SrcB will only have its first face unpacked.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    constexpr bool swap_operands          = reduce_swaps_operands<POOL_TYPE, REDUCE_DIMENSION>();
    constexpr std::uint32_t data_unpacker = swap_operands ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    constexpr std::uint32_t clr_mode      = (POOL_TYPE == PoolType::MAX) ? p_unpacr::UNP_CLRSRC_NEGINF : p_unpacr::UNP_CLRSRC_ZERO;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    std::uint32_t unpack_data_face;
    std::uint32_t unpack_scaler;
    if (num_faces == NUM_FACES)
    {
        unpack_data_face = swap_operands ? TT_OP_UNPACR1_FACE_INC(0, 1 /*Src Face Idx*/, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/)
                                         : TT_OP_UNPACR0_FACE_INC(0, 1 /*Src Face Idx*/, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_scaler    = swap_operands ? TT_OP_UNPACR0_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/)
                                         : TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_data_face = swap_operands ? TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/)
                                         : TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_scaler =
            swap_operands ? TT_OP_UNPACR0_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/) : TT_OP_UNPACR1_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }

    if (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR || tensor_shape.face_r_dim < MAX_FACE_R_DIM)
    {
        const std::uint32_t clear_data_bank =
            TT_OP_UNPACR_NOP(data_unpacker, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);

        ckernel_template temp(num_tiles, num_faces, clear_data_bank, unpack_data_face);
        temp.set_start_op(unpack_scaler);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(num_tiles, num_faces, unpack_data_face);
        temp.set_start_op(unpack_scaler);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker for reduce operations.
 *
 * Validates the tensor shape, programs the SrcA transpose for REDUCE_ROW, then programs the MOP for
 * unpacking a full tile for SrcA and a single face for SrcB. buf_desc_id_0 feeds UNPACKER0 -> SRCA,
 * buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 * For REDUCE_ROW SUM/AVG the operands are swapped (see @ref reduce_swaps_operands).
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @param num_tiles: Number of tiles to unpack at a time for SrcA; SrcB will only have its first face unpacked.
 * @note On the math thread, pair with @ref _llk_math_reduce_init_ (T1); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_unpack_reduce_ is the matching execute call on this thread.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape tensor_shape, const std::uint32_t num_tiles = NUM_TILES)
{
    static_assert(
        POOL_TYPE != PoolType::MIN,
        "The FPU reduce has no MIN: the unpacker's partial-face fill clears SrcA to zero for every "
        "pool but MAX, which a min reduce would lose to. Use the SFPU reduce instead "
        "(ckernel_sfpu_reduce.h::calculate_reduce).");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW));
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_reduce_mop_config_<POOL_TYPE, REDUCE_DIMENSION>(buf_desc_id_0, buf_desc_id_1, tensor_shape, num_tiles);
}

/**
 * @brief Unpacks operands for reduce kernels into SrcA and SrcB.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER0 -> SRCA, start_l1_tile_idx_1 -> UNPACKER1 -> SRCB.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @note Call @ref _llk_unpack_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    constexpr bool swap_operands = reduce_swaps_operands<POOL_TYPE, REDUCE_DIMENSION>();

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset

    const std::uint32_t hw_tiles_per_sw_tile = (tensor_shape.total_num_faces() == NUM_FACES) ? 1 : tensor_shape.total_num_faces();
    const std::uint32_t l1_idx_data          = start_l1_tile_idx_0 * hw_tiles_per_sw_tile;
    const std::uint32_t l1_idx_scaler        = start_l1_tile_idx_1 * hw_tiles_per_sw_tile;

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, swap_operands ? l1_idx_scaler : l1_idx_data);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, swap_operands ? l1_idx_data : l1_idx_scaler);

    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
