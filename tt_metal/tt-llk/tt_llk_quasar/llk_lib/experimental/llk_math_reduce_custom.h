// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// Pool slice length: an SrcA completion fence plus GMPOOL for each face in a pair.
constexpr std::uint32_t REDUCE_MAX_ROW_POOL_LEN = 4;
// Transpose slice starts right after the pool slice.
constexpr std::uint32_t REDUCE_MAX_ROW_TRANS_START = REDUCE_MAX_ROW_POOL_LEN;

// Length of the recorded 16-bit transpose slice for this tensor shape. Must match between the
// recording (mop_config) and the execute-side replay. Per face-row: MOVD2B(transpose) +
// MOVD2B(copy) + ZEROSRC + ELWADDDI [+ ELWADDDI when face_r_dim > 8]; plus a leading SETRWC
// (cr_d=0), an inter-face SETRWC (+32) when there are two face-rows, and a final SETRWC.
inline std::uint32_t _reduce_max_row_trans_len_(const ckernel::TensorShape& tensor_shape)
{
    const std::uint32_t writeback_instructions = (tensor_shape.face_r_dim + ELTWISE_MATH_ROWS - 1) / ELTWISE_MATH_ROWS;
    const std::uint32_t one_row                = 3 + writeback_instructions;
    return 1 + one_row + (tensor_shape.total_num_faces() == NUM_FACES ? (1 + one_row) : 0) + 1;
}

template <std::uint32_t even_lreg, std::uint32_t odd_lreg, std::uint32_t rotations>
inline __attribute__((always_inline)) void _reduce_max_row_sfpu_rotate_pair_()
{
#pragma GCC unroll 8
    for (std::uint32_t i = 0; i < rotations; ++i)
    {
        // Mode 3 rotates one LREG across the eight Quasar SFPU columns. Global SFPSHFT2 has
        // two-cycle throughput, and Quasar v4.0 does not reliably insert the required bubble.
        TTI_SFPSHFT2(0, even_lreg, even_lreg, 3);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPSHFT2(0, odd_lreg, odd_lreg, 3);
        TTI_SFPNOP(0, 0, 0);
    }
}

/**
 * @brief Moves one FP32 pool result row into the first column of its destination face.
 *
 * A Quasar FP32 SFPLOAD covers four rows and either the eight even or eight odd columns. The pool
 * result occupies row zero, so the two loads below hold all 16 values across the eight SFPU
 * columns. Cross-column rotates bring four successive values to column zero; SFPTRANSP then maps
 * those four LREG row-zero values onto the four physical SFPU rows. Only column zero is meaningful
 * after each store, which is exactly the datum selected by the pack reduce mask.
 */
template <std::uint32_t face_base>
inline __attribute__((always_inline)) void _reduce_max_row_transpose_fp32_face_sfpu_();

template <std::uint32_t face_base, std::uint32_t output_addr, std::uint32_t first_rotations, std::uint32_t second_rotations>
inline __attribute__((always_inline)) void _reduce_max_row_transpose_fp32_four_rows_sfpu_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, face_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, face_base + 2);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG5, 0);
    _reduce_max_row_sfpu_rotate_pair_<p_sfpu::LREG0, p_sfpu::LREG1, first_rotations>();
    _reduce_max_row_sfpu_rotate_pair_<p_sfpu::LREG4, p_sfpu::LREG5, second_rotations>();
    // Quasar can drop the fourth transpose input when the cross-lane shifts target LREG3.
    // Shift in the independent second grid, then move the completed values into grid zero.
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG3, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG6, 0);
    TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG7, 0);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPTRANSP;
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, face_base + output_addr);
}

template <std::uint32_t face_base>
inline __attribute__((always_inline)) void _reduce_max_row_transpose_fp32_last_four_rows_sfpu_()
{
    _reduce_max_row_transpose_fp32_four_rows_sfpu_<face_base, 12, 2, 1>();
}

template <std::uint32_t face_base>
inline __attribute__((always_inline)) void _reduce_max_row_transpose_fp32_face_sfpu_()
{
    // Four passes map the sixteen source columns into successive groups of four output rows.
    _reduce_max_row_transpose_fp32_four_rows_sfpu_<face_base, 0, 0, 7>();
    _reduce_max_row_transpose_fp32_four_rows_sfpu_<face_base, 4, 6, 5>();
    _reduce_max_row_transpose_fp32_four_rows_sfpu_<face_base, 8, 4, 3>();
    _reduce_max_row_transpose_fp32_last_four_rows_sfpu_<face_base>();
}

inline void _reduce_max_row_transpose_fp32_sfpu_(const std::uint32_t dst_index)
{
    _llk_math_sfpu_start_(dst_index);
    _reduce_max_row_transpose_fp32_face_sfpu_<0>();
    _reduce_max_row_transpose_fp32_face_sfpu_<32>();
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _llk_math_sfpu_done_();
}

template <std::uint32_t face_base>
inline void _reduce_max_row_transpose_fp32_one_face_sfpu_(const std::uint32_t dst_index)
{
    _llk_math_sfpu_start_(dst_index);
    _reduce_max_row_transpose_fp32_face_sfpu_<face_base>();
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _llk_math_sfpu_done_();
}

/**
 * @brief Max-combines one face column from the following destination tile into the current tile.
 */
template <std::uint32_t face_base>
inline void _reduce_max_row_combine_fp32_face_tiles_sfpu_(const std::uint32_t dst_index)
{
    constexpr std::uint32_t next_tile_offset = MAX_TILE_R_DIM * 2;
    _llk_math_sfpu_start_(dst_index);

    const auto combine_four_rows = [=](const std::uint32_t row_offset)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, face_base + row_offset);
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, next_tile_offset + face_base + row_offset);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, face_base + row_offset);
    };

    combine_four_rows(0);
    combine_four_rows(4);
    combine_four_rows(8);
    combine_four_rows(12);

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _llk_math_sfpu_done_();
}

/**
 * @brief Combines two already row-reduced FP32 tiles, keeping the maximum in the first tile.
 */
inline void _reduce_max_row_combine_fp32_tiles_sfpu_(const std::uint32_t dst_index)
{
    constexpr std::uint32_t next_tile_offset = MAX_TILE_R_DIM * 2;
    _llk_math_sfpu_start_(dst_index);

    const auto combine_four_rows = [=](const std::uint32_t row_offset)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, row_offset);
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, next_tile_offset + row_offset);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, row_offset);
    };

    combine_four_rows(0);
    combine_four_rows(4);
    combine_four_rows(8);
    combine_four_rows(12);
    combine_four_rows(MAX_TILE_R_DIM);
    combine_four_rows(MAX_TILE_R_DIM + 4);
    combine_four_rows(MAX_TILE_R_DIM + 8);
    combine_four_rows(MAX_TILE_R_DIM + 12);

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _llk_math_sfpu_done_();
}

/**
 * @brief Combines four native 16x32 BF16 row reductions into the first destination tile.
 */
inline void _reduce_max_row_combine_fp16b_tiles_sfpu_(const std::uint32_t dst_index)
{
    _set_dst_write_addr_by_rows_(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);

    // Follow the canonical Quasar SFPU walk. Each iteration max-combines the
    // same two physical rows from all four tile slots, stores once, then advances.
#pragma GCC unroll 8
    for (std::uint32_t row_group = 0; row_group < FACE_R_DIM / SFP_ROWS; ++row_group)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 32);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 64);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 96);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0, 0, SFP_ROWS, 0>();
    }

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _llk_math_sfpu_done_();
}

/**
 * @brief Native 16x32 BF16 row max without MOP replay.
 *
 * The repeated generic row-reduce MOP can stop producing after several fused block sections on
 * Quasar. This is its two-face instruction sequence with explicit SrcA fences and dvalid release.
 */
inline void _llk_math_reduce_row_fp16b_tile_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
    TTI_ELWADDDI(p_elwise::CLR_NONE, 0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0, ADDR_MOD_1, 0);
    TTI_ELWADDDI(p_elwise::CLR_NONE, 0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0, ADDR_MOD_1, 0);
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Sets up addrmods for block reduce_max_row (pool + transpose writeback).
 *
 * ADDR_MOD_0: pool + 16-bit transpose reads, no counter auto-increment.
 * ADDR_MOD_1: 16-bit transpose writeback, advances SrcB and DEST by 8 rows.
 * ADDR_MOD_7: SFPU FP32 load/store, no address auto-increment.
 */
inline void reduce_max_row_configure_addrmod()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = ELTWISE_MATH_ROWS},
        .dest = {.incr = ELTWISE_MATH_ROWS},
    }
        .set(ADDR_MOD_1);

    _sfpu_configure_addrmod_();
}

/**
 * @brief Minimal addrmod re-set after a fused matmul/eltwise clobbered ADDR_MOD_0/1.
 */
inline void reduce_max_row_configure_addrmod_reinit_minimal()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = ELTWISE_MATH_ROWS},
        .dest = {.incr = ELTWISE_MATH_ROWS},
    }
        .set(ADDR_MOD_1);

    _sfpu_configure_addrmod_();
}

/**
 * @brief Pools one input face into the selected FP32 DEST row and releases SrcA.
 */
template <std::uint32_t dst_row>
inline void _llk_math_reduce_row_fp32_face_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, dst_row>();
}

/**
 * @brief Records the block-reduce replay buffer and programs the MOP.
 *
 * The MOP loop only pools; GMPOOL max-accumulates every block tile into fixed DEST rows (row 0 for
 * F0&F1, row 32 for F2&F3), so the transpose runs once after the whole block. Both destination modes
 * record the transpose after the two pool instructions.
 *
 * @tparam block_ct_dim: number of operand tiles in the block (MOP outer-loop count).
 * @tparam is_fp32_dest_acc_en: selects the 32-bit dest transpose path in the execute function.
 * @param tensor_shape: operand tile shape (num faces, face dims).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape& tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    const std::uint32_t replay_buf_len = REDUCE_MAX_ROW_POOL_LEN + (is_fp32_dest_acc_en ? 0 : _reduce_max_row_trans_len_(tensor_shape));

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tensor_shape]
        {
            // ---- POOL SLICE: pool one face-pair into DEST[cr_d], max-accumulating over the block ----
            // A SrcA-valid fence adds the cycle needed for the final Float32 unpack writes to land
            // before each GMPOOL reads the face. The first pool releases SrcA valid so the paired
            // unpack MOP streams the next face; the second is CLR_NONE.
            TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
            tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
            TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
            tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            if constexpr (!is_fp32_dest_acc_en)
            {
                const std::uint32_t writeback_instructions = (tensor_shape.face_r_dim + ELTWISE_MATH_ROWS - 1) / ELTWISE_MATH_ROWS;

                // ---- TRANSPOSE SLICE: transpose both accumulated face-rows to 16x1 columns, once per block ----
                // Face-row 0 (DEST rows 0-15):
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB); // cr_d = 0
                TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON,
                           0); // read DEST, transpose into SrcB
                TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
                TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
                for (std::uint32_t i = 0; i < writeback_instructions; ++i)
                {
                    TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
                }

                if (tensor_shape.total_num_faces() == NUM_FACES)
                {
                    // Face-row 2 (DEST rows 32-47):
                    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, MAX_TILE_R_DIM, p_setrwc::SET_D); // cr_d = 32
                    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
                    TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
                    for (std::uint32_t i = 0; i < writeback_instructions; ++i)
                    {
                        TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
                    }
                }

                TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD); // final reset
            }
        });

    // Pool slice replayed for one face-pair; NOT the transpose (that runs once from execute).
    const std::uint32_t pool_replay = TT_OP_REPLAY(0, REDUCE_MAX_ROW_POOL_LEN, 0, 0, 0, 0);

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        // Per tile: pool F0&F1 @cr_d=0 -> inner SETRWC(+32) -> pool F2&F3 @cr_d=32 -> release SrcA + reset.
        // +32 lives in the inner loop op so it fires on every outer iteration, including the last.
        ckernel_template temp(block_ct_dim, 1, TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, MAX_TILE_R_DIM, p_setrwc::SET_D));
        temp.set_start_op(pool_replay);
        temp.set_end_ops(pool_replay, TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_ABD_F));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        // Single face-row (16x32 tiny tile): pool F0&F1 only, no F2 jump.
        ckernel_template temp(block_ct_dim, 1, TT_OP_NOP);
        temp.set_start_op(pool_replay);
        temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_ABD_F));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Reprograms only the block-reduce MOP registers, leaving the recorded replay buffer intact.
 *
 * Use when a fused eltwise/matmul clobbered the MOP config but the replay buffer at slots
 * [0, POOL_LEN+TRANS_LEN) still holds the pool + transpose sequence.
 */
template <std::uint32_t block_ct_dim>
inline void _llk_math_reduce_block_max_row_mop_reprogram_only_(const ckernel::TensorShape& tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    const std::uint32_t pool_replay = TT_OP_REPLAY(0, REDUCE_MAX_ROW_POOL_LEN, 0, 0, 0, 0);

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        ckernel_template temp(block_ct_dim, 1, TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, MAX_TILE_R_DIM, p_setrwc::SET_D));
        temp.set_start_op(pool_replay);
        temp.set_end_ops(pool_replay, TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_ABD_F));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(block_ct_dim, 1, TT_OP_NOP);
        temp.set_start_op(pool_replay);
        temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_ABD_F));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes addrmods, dest tile shape, counters and the MOP for block reduce_max_row.
 *
 * @tparam block_ct_dim: number of operand tiles in the block.
 * @tparam is_fp32_dest_acc_en: selects the 32-bit dest transpose path.
 * @param tensor_shape: operand tile shape.
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_block_max_row_init_ (T0).
 * @note @ref _llk_math_reduce_block_max_row_ runs the configured reduction with matching template args.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    reduce_max_row_configure_addrmod();
    _init_sfpu_config_reg_();
    if constexpr (is_fp32_dest_acc_en)
    {
        LLK_ASSERT(tensor_shape.total_num_faces() == NUM_FACES, "FP32 block row max requires a 32x32 tile");
        LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "FP32 block row max requires full-height faces");
    }
    else if (tensor_shape.num_faces_r_dim == 1)
    {
        static_assert(block_ct_dim == 4, "16x32 block row max currently supports four tiles per block");
    }
    else
    {
        _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, false>(tensor_shape);
    }
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Block reduce_max_row uninit (no state to restore).
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
}

/**
 * @brief Runs block reduce_max_row: pool the whole block into DEST, then transpose once.
 *
 * @tparam block_ct_dim: number of operand tiles in the block.
 * @tparam is_fp32_dest_acc_en: selects the 32-bit dest transpose path.
 * @param dst_index: destination tile slot.
 * @param tensor_shape: operand tile shape.
 * @note Call @ref _llk_math_reduce_block_max_row_init_ with matching template args before this function.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    if constexpr (is_fp32_dest_acc_en)
    {
        static_assert(block_ct_dim == 2, "FP32 block row max currently supports two tiles per block");

        // Every face is independently pooled and transposed. The following destination tile is
        // scratch, so all maxima are lane-wise between identically laid-out FP32 face columns.
        //
        // First input tile, F0: initialize the top output face.
        _set_dst_write_addr_by_rows_(dst_index);
        _llk_math_reduce_row_fp32_face_<0>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<0>(dst_index);

        // First input tile, F1: transpose in scratch and combine into the top output face.
        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<0>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<0>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<0>(dst_index);

        // First input tile, F2/F3: initialize then combine the bottom output face.
        _set_dst_write_addr_by_rows_(dst_index);
        _llk_math_reduce_row_fp32_face_<MAX_TILE_R_DIM>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<MAX_TILE_R_DIM>(dst_index);

        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<MAX_TILE_R_DIM>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<MAX_TILE_R_DIM>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<MAX_TILE_R_DIM>(dst_index);
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);

        // Second input tile: every face is reduced through scratch and combined into its matching
        // output face. This completes the 64-column block maximum without a 32-bit Dest round-trip.
        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<0>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<0>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<0>(dst_index);

        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<0>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<0>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<0>(dst_index);

        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<MAX_TILE_R_DIM>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<MAX_TILE_R_DIM>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<MAX_TILE_R_DIM>(dst_index);

        _set_dst_write_addr_by_rows_(dst_index + 1);
        _llk_math_reduce_row_fp32_face_<MAX_TILE_R_DIM>();
        _reduce_max_row_transpose_fp32_one_face_sfpu_<MAX_TILE_R_DIM>(dst_index + 1);
        _reduce_max_row_combine_fp32_face_tiles_sfpu_<MAX_TILE_R_DIM>(dst_index);
    }
    else if (tensor_shape.num_faces_r_dim == 1)
    {
        // Preserve each native 16x32 row reduction in its own destination slot, then combine the
        // four columns in SFPU. This avoids GMPOOL cross-tile destination accumulation.
        for (std::uint32_t tile = 0; tile < block_ct_dim; ++tile)
        {
            _set_dst_write_addr_by_rows_(dst_index + tile);
            TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);
            _llk_math_reduce_row_fp16b_tile_();
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
        }
        _reduce_max_row_combine_fp16b_tiles_sfpu_(dst_index);
    }
    else
    {
        _set_dst_write_addr_by_rows_(dst_index);

        // Pool all block_ct_dim tiles into DEST rows 0 (F0&F1) and 32 (F2&F3).
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

        // SrcB dvalid handshake required before MOVD2B.
        TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);
        TT_REPLAY(REDUCE_MAX_ROW_TRANS_START, _reduce_max_row_trans_len_(tensor_shape), 0, 0, 0, 0);
    }

    // The tiny path releases SrcB once per logical input tile. Clearing it again
    // here can erase the next block's scaler dvalid after the unpacker has already
    // prefetched it, leaving math and unpack permanently out of phase.
    if (tensor_shape.num_faces_r_dim != 1)
    {
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
    }
}
