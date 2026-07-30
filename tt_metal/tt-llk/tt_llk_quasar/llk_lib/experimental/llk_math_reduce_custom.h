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

// F2/F3 reduce into a separate row-form accumulator at DEST row 32.
static constexpr std::uint32_t REDUCE_BLOCK_SLOT1_DST = TILE_R_DIM;

/**
 * @brief Configure the FPU address modifiers used by block reduce_max_row.
 *
 * ADDR_MOD_0 keeps the GMPOOL and 32-bit move addresses fixed. ADDR_MOD_1
 * advances SrcB and DEST by eight rows for the native 16-bit transpose.
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
}

/**
 * @brief Restore address modifiers after another fused operation changes them.
 */
inline void reduce_max_row_configure_addrmod_reinit_minimal()
{
    reduce_max_row_configure_addrmod();
}

/**
 * @brief Transpose one FP32 GMPOOL result row using FPU move instructions.
 *
 * A Quasar Src register cannot hold a complete FP32 datum. Transpose the low
 * and high 16-bit halves independently, cache the low half in SrcA, then write
 * both halves back to DEST. The paired transposed/non-transposed MOVD2B
 * operations also retain the row-form maximum required by the next GMPOOL.
 */
inline void _reduce_max_row_transpose_fp32_fpu_()
{
    _configure_mov_ops_explicit_alu_data_format_state_<true>(DataFormat::Tf32, DataFormat::Tf32);
    _reduce_row_transpose_alu_cfg_enter_();

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);

    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);

    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 4);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 8);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 12);

    TTI_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0);
    TTI_MOVA2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 8);

    _reduce_row_transpose_alu_cfg_exit_();
    _configure_default_alu_data_format_state_<true /* IMPLIED_MATH_FORMAT */, true /* EN_32BIT_DEST */>(DataFormat::Tf32, DataFormat::Tf32);
}

/**
 * @brief Pool one face row from every FP32 tile in a horizontal block.
 *
 * The final pool keeps its SrcA token valid because the FP32 transpose helper
 * fills that bank with MOVB2A and consumes it with MOVA2D. All preceding pools
 * release SrcA so unpack can stream through the block.
 */
template <std::uint32_t block_ct_dim>
inline void _llk_math_reduce_row_fp32_block_face_row_()
{
    for (std::uint32_t tile = 0; tile < block_ct_dim; ++tile)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
        tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

        TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
        if (tile + 1 == block_ct_dim)
        {
            tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
        }
        else
        {
            tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
        }
    }
}

/**
 * @brief Configure the 16-bit pool-only block MOP.
 *
 * Quasar streams every transposed face through SrcA rows 0-15, alternating
 * banks. Each GMPOOL therefore consumes exactly one bank token with
 * CLR_SRCA_VLD. F0/F1 accumulate at DEST row 0 and F2/F3 at row 32. There is
 * deliberately no trailing SETRWC(CLR_A): that would release SrcA twice.
 */
template <std::uint32_t block_ct_dim>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape& tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");

    const bool two_face_rows     = (tensor_shape.num_faces_r_dim > 1);
    const std::uint32_t pool_len = two_face_rows ? 4u : 2u;

    load_replay_buf(
        0,
        pool_len,
        false,
        0,
        0,
        [two_face_rows]
        {
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS,
                       0); // F0 -> slot 0
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS,
                       0); // F1 -> slot 0
            if (two_face_rows)
            {
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS,
                           REDUCE_BLOCK_SLOT1_DST); // F2 -> slot 1
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS,
                           REDUCE_BLOCK_SLOT1_DST); // F3 -> slot 1
            }
        });

    const std::uint32_t pool_replay = TT_OP_REPLAY(0, pool_len, 0, 0, 0, 0);
    ckernel_template temp(1 /* outer */, block_ct_dim /* inner */, pool_replay);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Transpose one final 16-bit row accumulator into an output column.
 */
inline void _reduce_max_row_transpose_fp16b_face_(const bool wide_face)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, p_movd2b::TRANSPOSE_ON, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
    if (wide_face)
    {
        TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
    }
}

/**
 * @brief Initialize GMPOOL/FPU block reduce_max_row.
 *
 * The 16-bit path keeps GMPOOL results in row form for the complete horizontal
 * block, matching the working Quasar/Blackhole schedule, then transposes once.
 * FP32 streams one face row across the block at a time and transposes each
 * accumulated row once.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    reduce_max_row_configure_addrmod();
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    if constexpr (is_fp32_dest_acc_en)
    {
        static_assert(block_ct_dim == 2, "FP32 block row max currently supports two tiles per block");
        LLK_ASSERT(tensor_shape.total_num_faces() == NUM_FACES, "FP32 block row max requires a 32x32 tile");
        LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "FP32 block row max requires full-height faces");
    }
    else
    {
        LLK_ASSERT(tensor_shape.total_num_faces() == 2 || tensor_shape.total_num_faces() == NUM_FACES, "16-bit block row max requires a 16x32 or 32x32 tile");
        _llk_math_reduce_block_max_row_mop_config_<block_ct_dim>(tensor_shape);
    }

    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
}

/**
 * @brief Max-reduce all horizontal block tiles into one destination tile.
 *
 * Both paths stream SrcA through the two banks and hold one SrcB scaler token
 * for the complete horizontal block.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    _set_dst_write_addr_by_rows_(dst_index);

    if constexpr (is_fp32_dest_acc_en)
    {
        _llk_math_reduce_row_fp32_block_face_row_<block_ct_dim>();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
        _reduce_max_row_transpose_fp32_fpu_();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, REDUCE_BLOCK_SLOT1_DST, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);

        _llk_math_reduce_row_fp32_block_face_row_<block_ct_dim>();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
        _reduce_max_row_transpose_fp32_fpu_();
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);

        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
    }
    else
    {
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

        const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);
        const bool wide_face     = (tensor_shape.face_r_dim > ELTWISE_MATH_ROWS);
        _reduce_max_row_transpose_fp16b_face_(wide_face);
        if (two_face_rows)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, REDUCE_BLOCK_SLOT1_DST, p_setrwc::SET_D);
            _reduce_max_row_transpose_fp16b_face_(wide_face);
        }
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
    }
}
