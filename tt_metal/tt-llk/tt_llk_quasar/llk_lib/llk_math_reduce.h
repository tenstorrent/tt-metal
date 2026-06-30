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
 * @brief Enable SrcA/SrcB format override before reduce-row MOV/transpose glue.
 *
 * Quasar @ref _configure_mov_ops_explicit_alu_data_format_state_ already leaves override enabled for
 * MOV_OPS_EXPLICIT_FMT; this enter/exit pair scopes override to the MOV sequence (WH parity).
 * Zero_Flag_disabled_src is not exposed in Quasar alu_config_t (unlike WH).
 */
inline void _reduce_row_transpose_alu_cfg_enter_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_override_RMW, 1);
    cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_override_RMW, 1);
    // cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, 1);
    // cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, 0);
    cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 1);
    cfg_rmw(ALU_FORMAT_SPEC_REG_Dstacc_override_RMW, 1);
}

/**
 * @brief Disable SrcA/SrcB format override after reduce-row MOV/transpose glue.
 */
inline void _reduce_row_transpose_alu_cfg_exit_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_override_RMW, 0);
    cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_override_RMW, 0);
    // cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, 0);
    // cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, 1);
    cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 0);
    cfg_rmw(ALU_FORMAT_SPEC_REG_Dstacc_override_RMW, 0);
}

/**
 * @brief Int32 half-dest row transpose at an explicit dest row (row-reduce uses row 0).
 *
 * Required whenever reading/writing int32 dest datums: each 32-bit value is split across
 * DEST_NORM (hi16) and DEST_32B_LOW (lo16). A single MOVD2B cannot see the full word.
 */
inline void _reduce_row_transpose_fpu_(const std::uint32_t dest_addr = 0)
{
    tensix_sync();
    _configure_mov_ops_explicit_alu_data_format_state_<true>(DataFormat::Int32, DataFormat::Int32);
    _reduce_row_transpose_alu_cfg_enter_();

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);

    // Step 1: Read lo16 from dest into SrcB rows 16-31 and transpose.
    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, dest_addr);
    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, dest_addr);

    // Step 2: Cache transposed lo16 from SrcB rows 16-31 into SrcA rows 0-15.
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);

    // Step 3: Read hi16 from dest into SrcB rows 16-31 and transpose.
    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, dest_addr);
    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, dest_addr);

    // Step 4: Write transposed hi16 back to dest from SrcB rows 16-31.
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_addr + 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_addr + 4);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_addr + 8);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_addr + 12);

    // Step 5: Write cached lo16 from SrcA back to dest lo16 address space.
    TTI_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, dest_addr + 0);
    TTI_MOVA2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, dest_addr + 8);

    tensix_sync();
    _reduce_row_transpose_alu_cfg_exit_();
    _llk_math_srcAB_hw_configure_<false, false /*fp32_dest*/, true /*int32_dest*/>(DataFormat::Int32, DataFormat::Int32);
    tensix_sync();
}

/**
 * @brief Seed the int32 dest rows with the max-pool identity (most-negative int) before the first GMPOOL.
 *
 * The GMPOOL accumulates max against dest, so an unseeded 0 dest collapses all-negative rows to 0.
 *
 * Seeds via SrcB (not SrcA): ZEROSRC CLR_B fills SrcB with the int -inf pattern and MOVB2D broadcasts
 * it into the int32 dest (hi16 via DEST_NORM, lo16 via DEST_32B_LOW). SrcA must NOT be touched here --
 * it holds the unpacked face the very next GMPOOL pools, and clobbering it drops a lane (MAX comes out
 * too low). MAX's GMPOOL ignores SrcB, so overwriting SrcB is safe.
 */
inline void _reduce_int32_dest_init_min_for_max_(const std::uint32_t dst_addr)
{
    TTI_ZEROSRC(1, 1, 0, 1, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_B); // SrcB = int -inf (SrcA/face untouched)
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);

    tensix_sync();
    _configure_mov_ops_explicit_alu_data_format_state_<true>(DataFormat::Int32, DataFormat::Int32);

    TTI_MOVB2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, dst_addr + 0);
    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, dst_addr + 8);
    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, dst_addr + 0);
    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, dst_addr + 8);

    tensix_sync();
    _llk_math_srcAB_hw_configure_<false, false /*fp32_dest*/, true /*int32_dest*/>(DataFormat::Int32, DataFormat::Int32);
    tensix_sync();
}

/**
 * @brief Pool one pair of input faces (one output face row) into dest at an explicit row offset.
 *
 * Int32-dest reduce only (LoFi exact-integer accumulation; no fidelity multi-pass).
 */
template <PoolType POOL_TYPE, std::uint8_t DST_ADDR>
inline void _reduce_row_pool_face_pair_()
{
    if constexpr (POOL_TYPE == PoolType::MAX)
    {
        // Seed dest to the int max-pool identity; GMPOOL accumulates max against it (it does not
        // overwrite a 0 dest), so without this all-negative rows reduce to 0.
        _reduce_int32_dest_init_min_for_max_(DST_ADDR);
    }
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, DST_ADDR>();
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, DST_ADDR>();
}

/**
 * @brief Perform reduce-row at runtime for Int32 dest using FPU transpose glue (not replay buffer).
 */
template <PoolType POOL_TYPE>
inline void _llk_math_reduce_row_int32_fpu_(const TensorShape& tensor_shape)
{
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
 * @brief Perform reduce-scalar at runtime for Int32 dest.
 *
 * Flow: pool the faces -> 1x16 partial row at scratch -> MOVD2B(transpose) the row into a SrcB column
 * -> MOVB2A into SrcA -> final pool to dest[0]. The MOVD2B->MOVB2A load (not a math-side MOVD2A from
 * dest) is required so SrcA stays source-valid for the final pool: a plain MOVD2A moves data but does
 * not raise SrcA dvalid, and the pool then consumes nothing (yields 0). This mirrors the canonical
 * MOP path (@ref _llk_math_reduce_scalar_mop_config_), kept as straight-line TTI.
 *
 * @note MAX is exact (Int8-range). SUM/AVG is only exact while per-column partials stay within Int8
 * range; larger partials truncate when MOVD2B narrows the int32 dest row. See REDUCE_INT8_QUASAR_HANDOFF.md.
 */
template <PoolType POOL_TYPE>
inline void _llk_math_reduce_scalar_int32_fpu_(const TensorShape& tensor_shape)
{
    constexpr std::uint32_t scratch_dst_addr = 16;

    for (std::uint32_t face = 0; face < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face++)
    {
        tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();
    }
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();

    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, scratch_dst_addr);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 8);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, scratch_dst_addr);
    _llk_math_srcAB_hw_configure_<false, false /*fp32_dest*/, true /*int32_dest*/>(DataFormat::Int8, DataFormat::Int8);
    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
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
    constexpr std::uint32_t replay_buf_len = 2 + (RUN_FID_LOOPS ? (2 * NUM_FIDELITY_PHASES) : 0);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        []
        {
            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0x0>();

            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0x0>();
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

    temp.program_bank0_sw_cntl(instrn_buffer);
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
    constexpr std::uint32_t MOP_INNER_LOOP      = 1;
    // Replay buf max len is 32, NUM_FIDELITY_PHASES will be larger than 3, hypothetical limit of 19 + 12 = 31
    constexpr std::uint32_t replay_buf_len = 19 + (RUN_FID_LOOPS ? (4 * NUM_FIDELITY_PHASES) : 0);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tensor_shape]
        {
            // Each face is transposed in the unpacker, and then faces 0 & 1 are pooled together
            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

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
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);

            // Increment dest by 32
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 32, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);

            /////////////////////
            // Second face Row //
            /////////////////////
            // Each face is transposed in the unpacker, and then faces 0 & 1 are pooled together
            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            if constexpr (RUN_FID_LOOPS)
            {
                for (std::uint32_t fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            // This will clear AB counters to 0, and cr d is 32
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);

            // Src B can only transpose rows [16-31], and output them at [32-47]
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);

            // Required for accumulating on multiple tiles at a time, accumulation can only work
            // on row not column
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

            // Copy transposed rows in SrcB from [32 - 47] to dest rows [32 - 48]
            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            // Set counters back to 0
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

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
 */
template <ReduceDim REDUCE_DIMENSION, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_addrmod_()
{
    constexpr bool high_fidelity               = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;

    addr_mod_t {
        .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ((REDUCE_DIMENSION == ReduceDim::REDUCE_COL) ? 16 : 0)}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_0);

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
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @param en_int32_dest: When true for REDUCE_ROW/REDUCE_SCALAR, skip MOV-glue MOP and use FPU glue at execute time.
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_init_ (T0); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_init_(const TensorShape& tensor_shape, const bool en_int32_dest = false)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    _llk_math_reduce_addrmod_<REDUCE_DIMENSION, MATH_FIDELITY_TYPE>();

    const bool use_int32_fpu_glue = en_int32_dest && (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW || REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        _llk_math_reduce_col_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        if (!use_int32_fpu_glue)
        {
            _llk_math_reduce_row_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
        }
        else
        {
            // _reduce_row_transpose_warmup_(); // prime SrcB once for the FPU glue path
        }
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR)
    {
        if (!use_int32_fpu_glue)
        {
            _llk_math_reduce_scalar_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
        }
    }

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform a reduce operation.
 *
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param tensor_shape: Tile shape; required when en_int32_dest is true for REDUCE_ROW/REDUCE_SCALAR.
 * @param en_int32_dest: When true for REDUCE_ROW/REDUCE_SCALAR, runs the runtime FPU glue path instead of the MOP.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_(const std::uint32_t tile_idx, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE, const bool en_int32_dest = false)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    const bool use_int32_fpu_glue = en_int32_dest && (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW || REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        if (use_int32_fpu_glue)
        {
            _llk_math_reduce_row_int32_fpu_<POOL_TYPE>(tensor_shape);
        }
        else
        {
            ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        }
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR)
    {
        if (use_int32_fpu_glue)
        {
            _llk_math_reduce_scalar_int32_fpu_<POOL_TYPE>(tensor_shape);
        }
        else
        {
            ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }

    // Since only 1 face of srcB is used for constant values,
    // can clear data valid after all operations are done
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
