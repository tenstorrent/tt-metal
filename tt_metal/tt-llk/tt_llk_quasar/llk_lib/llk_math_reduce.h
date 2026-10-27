// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "llk_defs.h"
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

// The replays hold every fidelity phase a face can take; a face with fewer phases replays only the last ones.
constexpr static std::uint32_t REDUCE_MAX_FIDELITY_PHASES = math_fidelity_phase_count(MathFidelity::HiFi4);
constexpr static std::uint32_t REDUCE_ROW_GROUPS          = FACE_R_DIM / ELTWISE_MATH_ROWS;

/**
 * @brief Dest rows one face occupies; faces shorter than MAX_FPU_ROWS are padded to MAX_FPU_ROWS rows.
 *
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
constexpr std::uint32_t _reduce_dest_face_rows_(const TensorShape& tensor_shape)
{
    return std::max<std::uint32_t>(tensor_shape.face_r_dim, MAX_FPU_ROWS);
}

/**
 * @brief Sets up addrmods for REDUCE_ROW SUM/AVG.
 *
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_sum_addrmod_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    const std::uint32_t dest_row_pitch = _reduce_dest_face_rows_(tensor_shape) * tensor_shape.num_faces_c_dim;

    addr_mod_t {.srcb = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_0);
    addr_mod_t {.srcb = {.clr = 1}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_1);
    addr_mod_t {.srcb = {.clr = 1}, .fidelity = {.clr = 1}}.set(ADDR_MOD_2);
    addr_mod_t {.srcb = {.clr = 1}, .dest = {.incr = static_cast<std::uint16_t>(dest_row_pitch)}, .fidelity = {.clr = 1}}.set(ADDR_MOD_3);
}

/**
 * @brief Sets up mop config for reduce row SUM/AVG operations.
 *
 * For reduce row, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 32x1 column output (which means 2 output faces, 16x1 col each), each face is multiplied (MVMUL) by the transposed scaler,
 * Face 0 + Face 1 are accumulated together, and Face 2 & 3 are accumulated together.
 *
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_sum_mop_config_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t face_len = REDUCE_MAX_FIDELITY_PHASES * REDUCE_ROW_GROUPS;

    load_replay_buf<0, 2 * face_len>(
        []
        {
#pragma GCC unroll 3
            for (std::uint32_t phase = 1; phase < REDUCE_MAX_FIDELITY_PHASES; phase++)
            {
#pragma GCC unroll 4
                for (const auto row : fpu_row_offsets<FACE_R_DIM - ELTWISE_MATH_ROWS>())
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, row);
                }
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, FACE_R_DIM - ELTWISE_MATH_ROWS);
            }
#pragma GCC unroll 4
            for (const auto row : fpu_row_offsets<FACE_R_DIM - ELTWISE_MATH_ROWS>())
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, row);
            }
            TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_2, FACE_R_DIM - ELTWISE_MATH_ROWS);

#pragma GCC unroll 3
            for (std::uint32_t phase = 1; phase < REDUCE_MAX_FIDELITY_PHASES; phase++)
            {
#pragma GCC unroll 4
                for (const auto row : fpu_row_offsets<FACE_R_DIM - ELTWISE_MATH_ROWS>())
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, row);
                }
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, FACE_R_DIM - ELTWISE_MATH_ROWS);
            }
#pragma GCC unroll 4
            for (const auto row : fpu_row_offsets<FACE_R_DIM - ELTWISE_MATH_ROWS>())
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, row);
            }
            TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_3, FACE_R_DIM - ELTWISE_MATH_ROWS);
        });

    const std::uint32_t len              = fidelity.phase_count * REDUCE_ROW_GROUPS;
    const std::uint32_t face             = TT_OP_REPLAY(face_len - len, len, 0, 0, 0, 0);
    const std::uint32_t last_face_in_row = TT_OP_REPLAY(2 * face_len - len, len, 0, 0, 0, 0);
    ckernel_template temp = (tensor_shape.num_faces_c_dim == 1) ? ckernel_template(1 /*outer loop*/, tensor_shape.num_faces_r_dim, last_face_in_row)
                                                                : ckernel_template(1 /*outer loop*/, tensor_shape.num_faces_r_dim, face, last_face_in_row);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for REDUCE_ROW MAX.
 *
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_max_addrmod_(const TensorShape& tensor_shape)
{
    const std::uint32_t row_groups     = std::max<std::uint32_t>(tensor_shape.face_r_dim / ELTWISE_MATH_ROWS, 1);
    const std::uint32_t dest_row_pitch = _reduce_dest_face_rows_(tensor_shape) * tensor_shape.num_faces_c_dim;

    addr_mod_t {}.set(ADDR_MOD_0);
    addr_mod_t {.dest = {.incr = static_cast<std::uint16_t>(tensor_shape.num_faces_c_dim == 1 ? dest_row_pitch : 0)}}.set(ADDR_MOD_1);
    addr_mod_t {.dest = {.incr = static_cast<std::uint16_t>(dest_row_pitch)}}.set(ADDR_MOD_2);
    addr_mod_t {.dest = {.clr = 1}}.set(ADDR_MOD_3);
    addr_mod_t {.srcb = {.incr = ELTWISE_MATH_ROWS}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_4);
    addr_mod_t {.srcb = {.clr = 1}, .dest = {.incr = static_cast<std::uint16_t>(dest_row_pitch - (row_groups - 1) * ELTWISE_MATH_ROWS)}}.set(ADDR_MOD_5);
}

/**
 * @brief Sets up mop config for reduce row MAX operations.
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
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_max_mop_config_(const TensorShape& tensor_shape)
{
    load_replay_buf<0, 7 + REDUCE_ROW_GROUPS>(
        []
        {
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);

            // Src B can only transpose rows [16-31], and output them at [32-47]
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1 /*transpose*/, 0);

            // Required for accumulating on multiple tiles at a time, accumulation can only work
            // on row not column
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0 /*transpose*/, 0);

        // Copy transposed rows in SrcB from [32 - 47] to dest rows [0 - 16].
#pragma GCC unroll 4
            for (std::uint32_t row_group = 1; row_group < REDUCE_ROW_GROUPS; row_group++)
            {
                TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_4, 0);
            }
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_5, 0);
        });

    const std::uint32_t num_faces  = tensor_shape.total_num_faces();
    const std::uint32_t row_groups = std::max<std::uint32_t>(tensor_shape.face_r_dim / ELTWISE_MATH_ROWS, 1);

    ckernel_template temp(
        1 /*outer loop*/,
        tensor_shape.num_faces_r_dim,
        TT_OP_REPLAY(5, 2, 0, 0, 0, 0),
        TT_OP_REPLAY(7 + REDUCE_ROW_GROUPS - row_groups, row_groups, 0, 0, 0, 0));
    temp.set_start_op(TT_OP_REPLAY(4 - num_faces, num_faces + 1, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_ABD_F));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for REDUCE_COL.
 *
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_col_addrmod_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    // For face_r_dim < 8, dest will be sparse with faces placed every 8 rows.
    // For face_r_dim >= 8, dest in a dense manner faces placed every tensor_shape.face_r_dim rows
    const std::uint16_t dest_face_incr = static_cast<std::uint16_t>((tensor_shape.face_r_dim < (FACE_R_DIM >> 1)) ? MAX_FPU_ROWS : tensor_shape.face_r_dim);

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = dest_face_incr}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);
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
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE>
inline void _llk_math_reduce_col_mop_config_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    // So Face 0 reduce, dest counter += 16, Face 1 reduce, dest counter reset to 0
    // then Face 2 reduce (which includes Face 0 reduce result in dest), dest counter += 16, Face 3 reduce(which includes Face 1 reduce result in dest at index
    // 16)
    const std::uint32_t MOP_OUTER_LOOP      = 1;
    const std::uint32_t MOP_INNER_LOOP      = (tensor_shape.total_num_faces() >= 2) ? (tensor_shape.total_num_faces() >> 1) : tensor_shape.total_num_faces();
    const std::uint32_t NUM_FIDELITY_PHASES = fidelity.phase_count - 1;
    const std::uint32_t replay_buf_len      = 2 + 2 * NUM_FIDELITY_PHASES;

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [NUM_FIDELITY_PHASES]
        {
            // <<< Starting Point = 0: For num_faces > 1 && num_faces_r_dim <= num_faces_c_dim >>> //
            for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
            {
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
            }
            // This increments dest by 8 rows for face_r_dim <= 8, and 16 otherwise
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0x0>();

            // <<< Starting Point = NUM_FIDELITY_PHASES + 1: For total_num_faces() == 1 || num_faces_c_dim < num_faces_r_dim >>> //
            for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
            {
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
            }
            // This resets dest for accumulation
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0x0>();
        });

    const std::uint32_t pool_one_face_and_reset_dest_adc      = TT_OP_REPLAY(replay_buf_len >> 1, replay_buf_len >> 1, 0, 0, 0, 0);
    const std::uint32_t pool_two_faces_and_ping_pong_dest_adc = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);

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
 * @brief Sets up addrmods for REDUCE_SCALAR.
 *
 * @param fidelity: Phase schedule for the original source operands.
 * @param final_fidelity: Phase schedule for the final pool, whose SrcA is copied back from dest.
 */
inline void _llk_math_reduce_scalar_addrmod_(const MathFidelitySchedule fidelity, const MathFidelitySchedule final_fidelity)
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t {.fidelity = {.incr = final_fidelity.phase_increment}}.set(ADDR_MOD_1);
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);
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
 * @tparam EN_32BIT_DEST: Use the active destination addressing mode when clearing the scratch row.
 * @param fidelity: Phase schedule for the original source operands.
 * @param final_fidelity: Phase schedule for the final pool, whose SrcA is copied back from dest.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, bool EN_32BIT_DEST>
inline void _llk_math_reduce_scalar_mop_config_(const MathFidelitySchedule fidelity, const MathFidelitySchedule final_fidelity, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t MOP_OUTER_LOOP    = 1;
    constexpr std::uint32_t MOP_INNER_LOOP    = 1;
    const std::uint32_t NUM_FIDELITY_PHASES   = fidelity.phase_count - 1;
    const std::uint32_t FINAL_FIDELITY_PHASES = final_fidelity.phase_count - 1;
    constexpr std::uint32_t b2a_moves         = FACE_R_DIM / ELTWISE_MATH_ROWS;
    const std::uint32_t replay_buf_len =
        4 + b2a_moves + tensor_shape.total_num_faces() - 1 + tensor_shape.total_num_faces() * NUM_FIDELITY_PHASES + FINAL_FIDELITY_PHASES;

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tensor_shape, NUM_FIDELITY_PHASES, FINAL_FIDELITY_PHASES]
        {
            // Set up a dest addr to output temp results into, has to be less than 16 (to not write into next tile)
            // but also has to be greater than 0 (where results are expected)
            constexpr std::uint32_t scratch_dst_addr = 8;

            // Pool all faces together (default 4 faces), this will generate 1x16 row of result at dst index scratch_dst_addr
            // No src/dest counters are incremented
            for (std::uint32_t face = 0; face < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face++)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
                }
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();
            }

            for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
            {
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();

            // Only SrcB transpose can be done from Math
            // Rows [16:31] of SrcB are transposed, then written back into rows [32:47] of SrcB
            // Following will move 1x16 pool result to SrcB to be transposed into 16 rows
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, scratch_dst_addr);

#pragma GCC unroll 4
            for (const auto row : fpu_row_offsets<FACE_R_DIM>())
            {
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + row, ADDR_MOD_0, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET + row);
            }

            // zero out scratch in dest
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, EN_32BIT_DEST, 0, ADDR_MOD_0, scratch_dst_addr);

            for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < FINAL_FIDELITY_PHASES; fid_phase_idx++)
            {
                // Do final pool of the transposed rows to generate a single pool datum
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0>();
            }
            // Do final pool of the transposed rows to generate a single pool datum
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods and mop config (initialization) for reduce operations.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam EN_32BIT_DEST: Set to true when destination registers use 32-bit addressing.
 * @tparam MATH_FIDELITY_TYPE: Requested multiplication fidelity for SUM/AVG; MAX always runs LoFi, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param src_a_format: Effective SrcA register format for the data operand (scaler for REDUCE_ROW SUM/AVG).
 * @param src_b_format: Effective SrcB register format for the scaler operand (data for REDUCE_ROW SUM/AVG).
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_init_ (T0); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 * @note Pass the source register formats: for SUM/AVG, init lowers the fidelity to the highest one they can use and skips phases they
 *       cannot use. Run init again whenever either format changes.
 * @note PoolType::MIN is rejected here. Nothing reduces without this init, so that closes the whole
 *       FPU path to it.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, bool EN_32BIT_DEST, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_init_(const DataFormat src_a_format, const DataFormat src_b_format, const TensorShape tensor_shape)
{
    constexpr MathFidelity FIDELITY = POOL_TYPE == PoolType::MAX ? MathFidelity::LoFi : MATH_FIDELITY_TYPE;
    // There is no min-pool instruction - the FPU has GMPOOL (max) and GAPOOL (average/sum) - so MIN
    // would not fail here, it would quietly average, and the unpacker would pad with zero where a
    // min reduce needs +inf. PoolType::MIN exists for the SFPU reduce, which implements it.
    static_assert(
        POOL_TYPE != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(
        src_a_format != DataFormat::Int8 || (tensor_shape.face_r_dim == FACE_R_DIM && tensor_shape.total_num_faces() == NUM_FACES),
        "Int8 reduce only supports 32x32 tiles");
    [[maybe_unused]] const auto fidelity = math_fidelity_schedule<FIDELITY>(src_a_format, src_b_format);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW && POOL_TYPE == PoolType::MAX)
    {
        _llk_math_reduce_row_max_addrmod_(tensor_shape);
        _llk_math_reduce_row_max_mop_config_(tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        _llk_math_reduce_row_sum_addrmod_(fidelity, tensor_shape);
        _llk_math_reduce_row_sum_mop_config_(fidelity, tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        _llk_math_reduce_col_addrmod_(fidelity, tensor_shape);
        _llk_math_reduce_col_mop_config_<POOL_TYPE>(fidelity, tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR)
    {
        const auto final_fidelity = math_fidelity_schedule<FIDELITY>(dest_src_format<EN_32BIT_DEST>(src_a_format), src_b_format);
        _llk_math_reduce_scalar_addrmod_(fidelity, final_fidelity);
        _llk_math_reduce_scalar_mop_config_<POOL_TYPE, EN_32BIT_DEST>(fidelity, final_fidelity, tensor_shape);
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
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param tensor_shape: Tile shape.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_math_reduce_(const std::uint32_t tile_idx, [[maybe_unused]] const TensorShape& tensor_shape)
{
    _set_dst_write_addr_by_rows_(tile_idx);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    constexpr std::uint32_t clear_scaler = reduce_swaps_operands<POOL_TYPE, REDUCE_DIMENSION>() ? p_setrwc::CLR_A : p_setrwc::CLR_B;
    TTI_SETRWC(clear_scaler, 0, 0, p_setrwc::SET_ABD_F);
}
