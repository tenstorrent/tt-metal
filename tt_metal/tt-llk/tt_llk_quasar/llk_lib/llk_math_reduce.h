// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Emit the pool instruction (GMPOOL for MAX, GAPOOL for SUM/AVG) matching the reduce pool type.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam CLR_SRC: Source-clear mode applied after pooling (p_gpool CLR_* value)
 * @tparam POOL_SIZE: Pool face dimension (p_gpool DIM_* value)
 * @tparam ADDR_MOD: Address-mod slot used by the instruction
 * @tparam MAX_POOL_IDX_EN: Enable max-pool index output (p_gpool INDEX_* value)
 * @tparam DST_ADDR: Destination write address
 */
template <PoolType POOL_TYPE, std::uint8_t CLR_SRC, std::uint8_t POOL_SIZE, std::uint8_t ADDR_MOD, std::uint8_t MAX_POOL_IDX_EN, std::uint8_t DST_ADDR>
void tti_pool_instr_func()
{
    if constexpr (POOL_TYPE == PoolType::MAX)
    {
        TTI_GMPOOL(CLR_SRC, POOL_SIZE, ADDR_MOD, MAX_POOL_IDX_EN, DST_ADDR);
    }
    else
    {
        TTI_GAPOOL(CLR_SRC, POOL_SIZE, ADDR_MOD, MAX_POOL_IDX_EN, DST_ADDR);
    }
}

/**
 * @brief Scope dest-format override around the reduce-row transpose.
 */
inline void _reduce_row_transpose_alu_cfg_enter_()
{
    constexpr std::uint8_t dstacc_override_rmw_b3_mask = static_cast<std::uint8_t>(1u << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT % 8));

    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    TTI_RMWCIB3(ALU_FORMAT_SPEC_REG_Dstacc_override_ADDR32, dstacc_override_rmw_b3_mask, dstacc_override_rmw_b3_mask);
}

/**
 * @brief Disable dest-format override after reduce-row transpose.
 */
inline void _reduce_row_transpose_alu_cfg_exit_()
{
    constexpr std::uint8_t dstacc_override_rmw_b3_mask = static_cast<std::uint8_t>(1u << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT % 8));

    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    TTI_RMWCIB3(ALU_FORMAT_SPEC_REG_Dstacc_override_ADDR32, dstacc_override_rmw_b3_mask, 0);
}

/**
 * @brief Int32 half-dest row transpose at dest row 0 (row-reduce result row).
 *
 * Required whenever reading/writing int32 dest datums: each 32-bit value is split across
 * DEST_NORM (hi16) and DEST_32B_LOW (lo16). A single MOVD2B cannot see the full word.
 *
 */
inline void _reduce_row_transpose_fpu_()
{
    _configure_mov_ops_explicit_alu_data_format_state_<true>(DataFormat::Int32, DataFormat::Int32);
    _reduce_row_transpose_alu_cfg_enter_();

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);

    // Step 1: Read lo16 from dest into SrcB rows 16-31 and transpose.
    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

    // Step 2: Cache transposed lo16 from SrcB rows 16-31 into SrcA rows 0-15.
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);

    // Step 3: Read hi16 from dest into SrcB rows 16-31 and transpose.
    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

    // Step 4: Write transposed hi16 back to dest from SrcB rows 16-31.
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 4);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 8);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 12);

    // Step 5: Write cached lo16 from SrcA back to dest lo16 address space.
    TTI_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0);
    TTI_MOVA2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 8);

    _reduce_row_transpose_alu_cfg_exit_();
    _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, true /* EN_32BIT_DEST */>(DataFormat::Int8, DataFormat::Int8);
}

/**
 * @brief Pool one pair of input faces (one output face row) into dest at an explicit row offset.
 *
 */
template <PoolType POOL_TYPE, std::uint8_t DST_ADDR>
inline void _reduce_row_pool_face_pair_()
{
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, DST_ADDR>();
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, DST_ADDR>();
}

/**
 * @brief Perform reduce-row at runtime for Int32 dest using FPU transpose.
 *
 * Full 32x32 tiles only — tiny tiles are not supported for Int8→Int32 reduce.
 */
template <PoolType POOL_TYPE>
inline void _llk_math_reduce_row_int32_fpu_(const TensorShape& tensor_shape)
{
    LLK_ASSERT(
        tensor_shape.face_r_dim == DEFAULT_TENSOR_SHAPE.face_r_dim && tensor_shape.face_c_dim == DEFAULT_TENSOR_SHAPE.face_c_dim &&
            tensor_shape.num_faces_r_dim == DEFAULT_TENSOR_SHAPE.num_faces_r_dim && tensor_shape.num_faces_c_dim == DEFAULT_TENSOR_SHAPE.num_faces_c_dim,
        "Int8 reduce-row: tiny tiles not supported (requires 32x32)");

    _reduce_row_pool_face_pair_<POOL_TYPE, 0>();
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    _reduce_row_transpose_fpu_();

    if (tensor_shape.num_faces_r_dim > 1)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 32, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);

        _reduce_row_pool_face_pair_<POOL_TYPE, 0>();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
        _reduce_row_transpose_fpu_();
    }

    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Sets up mop config for reduce column operations.
 *
 * For reduce Col, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 1x32 row output (which means 2 output faces, 1x16 row each), then Face 0 + Face 2 are pooled together, and Face 1 and Face 3 are pooled
 * together.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types; sets how many loops to use full precision of Source register datums with multiplies, values =
 * <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_col_mop_config_(const TensorShape& tensor_shape)
{
    // So Face 0 reduce, dest counter += 16, Face 1 reduce, dest counter reset to 0
    // then Face 2 reduce (which includes Face 0 reduce result in dest), dest counter += 16, Face 3 reduce(which includes Face 1 reduce result in dest at index
    // 16)
    const std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP = (tensor_shape.total_num_faces() >= 2) ? (tensor_shape.total_num_faces() >> 1) : tensor_shape.total_num_faces();
    constexpr std::uint32_t NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : to_underlying(MATH_FIDELITY_TYPE) - 1;
    constexpr bool RUN_FID_LOOPS           = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));
    constexpr std::uint32_t replay_buf_len      = 2 + (RUN_FID_LOOPS ? 2 * NUM_FIDELITY_PHASES : 0);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        []
        {
            // <<< Starting Point = 0: For num_faces > 1 && num_faces_r_dim <= num_faces_c_dim >>> //
            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            // This increments dest by 8 rows for face_r_dim <= 8, and 16 otherwise
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0x0>();

            // <<< Starting Point = NUM_FIDELITY_PHASES + 1: For total_num_faces() == 1 || num_faces_c_dim < num_faces_r_dim >>> //
            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            // This resets dest for accumulation
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0x0>();
        });

    constexpr std::uint32_t pool_one_face_and_reset_dest_adc      = TT_OP_REPLAY(replay_buf_len >> 1, replay_buf_len >> 1, 0, 0, 0, 0);
    constexpr std::uint32_t pool_two_faces_and_ping_pong_dest_adc = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);

    if (tensor_shape.total_num_faces() == 1)
    {
        // Ensures only 1 pool instruction is issued for num_faces = 1 case.
        // Calls pool instruction with addr_mod_1 to ensure dest counters are reset at the end of instruction.
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pool_one_face_and_reset_dest_adc);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else if (tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim)
    {
        // If the tensor_shape is narrow, then there is only one column of faces. Both faces should be pooled to the same address.
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pool_one_face_and_reset_dest_adc, pool_one_face_and_reset_dest_adc);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        // In every other case, we should be incrementing dest_addr for every second face and resetting it for every first.
        // Run the entire MOP to get this functionality.
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pool_two_faces_and_ping_pong_dest_adc);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Sets up mop config for reduce row operations.
 *
 * For reduce row, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 32x1 column output (which means 2 output faces, 16x1 col each), then all faces are transposed, Face 0 + Face 1 are pooled together, and
 * Face 2 & 3 are pooled together.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types; sets how many loops to use full precision of Source register datums with multiplies, values =
 * <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_row_mop_config_(const TensorShape& tensor_shape)
{
    constexpr bool RUN_FID_LOOPS = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));
    constexpr std::uint32_t NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : to_underlying(MATH_FIDELITY_TYPE) - 1;
    constexpr std::uint32_t MOP_OUTER_LOOP      = 1;
    const std::uint32_t MOP_INNER_LOOP          = (tensor_shape.total_num_faces() >= 2 && !(tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim))
                                                      ? (tensor_shape.total_num_faces() >> 1)
                                                      : tensor_shape.total_num_faces();

    std::uint32_t replay_buf_len = 7 + NUM_FIDELITY_PHASES;
    if (tensor_shape.total_num_faces() > 1 && tensor_shape.num_faces_c_dim >= tensor_shape.num_faces_r_dim)
    {
        if (tensor_shape.total_num_faces() == NUM_FACES)
        {
            replay_buf_len++;
        }
        replay_buf_len += NUM_FIDELITY_PHASES + 1U;
    }

    if (tensor_shape.face_r_dim > ELTWISE_MATH_ROWS)
    {
        replay_buf_len++;
    }

    const std::uint32_t tail_len = 1U + (tensor_shape.total_num_faces() == NUM_FACES ? 1U : 0U);
    const std::uint32_t main_len = replay_buf_len - tail_len;

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tensor_shape]
        {
            // Each face is transposed in the unpacker, and then faces 0 & 1 are pooled together
            // 16x16 case (num_faces == 1) and 32x16 (narrow) case pools one face per row, all others pool
            // two faces per row. Skip this first pool for 16x16, 32x16.
            if (tensor_shape.total_num_faces() > 1 && tensor_shape.num_faces_c_dim >= tensor_shape.num_faces_r_dim)
            {
                if constexpr (RUN_FID_LOOPS)
                {
                    for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                    {
                        tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                    }
                }
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
            }

            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            // This will clear AB counters to 0, and cr d is also 0
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);

            // Src B can only transpose rows [16-31], and output them at [32-47]
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);

            // Required for accumulating on multiple tiles at a time, accumulation can only work
            // on row not column
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

            // Copy transposed rows in SrcB from [32 - 47] to dest rows [0 - 16]
            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);

            // For tiny-tiles, only the first 8 rows matter as they are the densely packed ones. We can skip the second copy in this case.
            if (tensor_shape.face_r_dim > ELTWISE_MATH_ROWS)
            {
                TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            }

            // For cases where each face is considered a tile, the dest counter is already aligned to 8 or 16.
            // Need to increment by 32 where all faces are considered a HW tile.
            if (tensor_shape.total_num_faces() == NUM_FACES)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, MAX_TILE_R_DIM, p_setrwc::SET_D);
            }
            TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);
        });

    const std::uint32_t replay           = TT_OP_REPLAY(0, main_len, 0, 0, 0, 0);
    const std::uint32_t dest_inc_32      = TT_OP_REPLAY(main_len, tail_len, 0, 0, 0, 0);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, replay, dest_inc_32);
    temp.set_last_inner_loop_instr(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up mop config for reduce scalar operations.
 *
 * For reduce scalar, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * All 4 faces will be pooled together; result will be a single reduce datum placed in datum 0 of the tile idx.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types; sets how many loops to use full precision of Source register datums with multiplies, values =
 * <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_scalar_mop_config_(const TensorShape& tensor_shape)
{
    constexpr std::uint32_t MOP_OUTER_LOOP      = 1;
    constexpr std::uint32_t MOP_INNER_LOOP      = 1;
    constexpr std::uint32_t NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : to_underlying(MATH_FIDELITY_TYPE) - 1;
    constexpr bool RUN_FID_LOOPS = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));
    const std::uint32_t replay_buf_len =
        6 + tensor_shape.total_num_faces() - 1 + (RUN_FID_LOOPS ? ((tensor_shape.total_num_faces() - 1) * NUM_FIDELITY_PHASES) + (2 * NUM_FIDELITY_PHASES) : 0);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tensor_shape]
        {
            // Set up a dest addr to output temp results into, has to be less than 64 (to not write into next tile)
            // but also has to be greater than 0 (where results are expected)
            constexpr std::uint32_t scratch_dst_addr = 16;

            // Pool all faces together (default 4 faces), this will generate 1x16 row of result at dst index scratch_dst_addr
            // No src/dest counters are incremented
            for (std::uint32_t face = 0; face < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face++)
            {
                if constexpr (RUN_FID_LOOPS)
                {
                    for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                    {
                        tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
                    }
                }
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();
            }

            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();

            // Only SrcB transpose can be done from Math
            // Rows [16:31] of SrcB are transposed, then written back into rows [32:47] of SrcB
            // Following will move 1x16 pool result to SrcB to be transposed into 16 rows
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, scratch_dst_addr);

            // copy over all 16 rows from B to A
            TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 0);
            TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 8);

            // zero out scratch in dest
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, scratch_dst_addr);

            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    // Do final pool of the transposed rows to generate a single pool datum
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            // Do final pool of the transposed rows to generate a single pool datum
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for reduce operations.
 *
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types; sets how many loops to use full precision of Source register datums with multiplies, values =
 * <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 */
template <ReduceDim REDUCE_DIMENSION, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_addrmod_(const TensorShape& tensor_shape)
{
    constexpr bool high_fidelity               = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;

    std::uint16_t addr_mod_0_dest_incr;
    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        if (tensor_shape.face_r_dim < (FACE_R_DIM >> 1))
        {
            // For face_r_dim < 8, dest will be sparse with faces placed every 8 rows.
            addr_mod_0_dest_incr = static_cast<std::uint16_t>(ELTWISE_MATH_ROWS);
        }
        else
        {
            // For face_r_dim >= 8, dest in a dense manner faces placed every tensor_shape.face_r_dim rows
            addr_mod_0_dest_incr = static_cast<std::uint16_t>(tensor_shape.face_r_dim);
        }
    }
    else
    {
        addr_mod_0_dest_incr = 0;
    }

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = addr_mod_0_dest_incr}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity_increment}}.set(ADDR_MOD_2);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = ELTWISE_MATH_ROWS},
            .dest = {.incr = ELTWISE_MATH_ROWS},
        }
            .set(ADDR_MOD_1);
    }
}

/**
 * @brief Sets up addrmods and mop config (initialization) for reduce operations.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types; sets how many loops to use full precision of Source register datums with multiplies, values =
 * <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: When true for REDUCE_ROW, skip MOP programming (runtime int FPU path).
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_init_ (T0); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, ckernel::MathFidelity MATH_FIDELITY_TYPE, bool is_int_fpu_en = false>
inline void _llk_math_reduce_init_(const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    _llk_math_reduce_addrmod_<REDUCE_DIMENSION, MATH_FIDELITY_TYPE>(tensor_shape);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        _llk_math_reduce_col_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        if constexpr (!is_int_fpu_en)
        {
            _llk_math_reduce_row_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
        }
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR)
    {
        _llk_math_reduce_scalar_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
    }

    // For face_r_dim >= 8, dest is dense with tiles. For face_r_dim < 8, dest is sparse with tiles and tiles are placed every 8 rows.
    // If num_rows_per_tile is less than that of face_r_dim = 8, replace it to ensure face_r_dim = 8 sparse layout.
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform a reduce operation.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_int_fpu_en: When true for REDUCE_ROW, runs the runtime int FPU path instead of the MOP.
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param tensor_shape: Tile shape; required when is_int_fpu_en is true for REDUCE_ROW.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const std::uint32_t tile_idx, const TensorShape& tensor_shape)
{
    static_assert(
        !(is_int_fpu_en && REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR && POOL_TYPE == PoolType::SUM),
        "Integer Scalar SUM/AVG (Int32 dest) unsupported on FPU: after the first GAPOOL, "
        "partials cannot fit back into Src for the final GAPOOL");

    _set_dst_write_addr_by_rows_(tile_idx);

    if constexpr (is_int_fpu_en && REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        _llk_math_reduce_row_int32_fpu_<POOL_TYPE>(tensor_shape);
    }
    else
    {
        // Non-int paths and the int32 scalar-MAX path both run the MOP configured in the init.
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }

    // Since only 1 face of srcB is used for constant values,
    // can clear data valid after all operations are done
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
