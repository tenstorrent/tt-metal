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
 * @brief Reduce one 32x32 FP32 tile into the current DEST tile.
 *
 * GMPOOL max-accumulates the top and bottom face pairs into arbitrary DEST
 * rows 0 and 32. Transposing after each tile preserves both the packed column
 * and the row-form accumulator needed by the following tile.
 */
inline void _llk_math_reduce_row_fp32_tile_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    _reduce_max_row_transpose_fp32_fpu_();

    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, MAX_TILE_R_DIM, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    tti_pool_instr_func<PoolType::MAX, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    _reduce_max_row_transpose_fp32_fpu_();

    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Initialize GMPOOL/FPU block reduce_max_row.
 *
 * FP32 uses the runtime split-half transpose above. BF16 reuses Quasar's
 * canonical 16x32 row-reduce MOP.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    reduce_max_row_configure_addrmod();
    if constexpr (is_fp32_dest_acc_en)
    {
        static_assert(block_ct_dim == 2, "FP32 block row max currently supports two tiles per block");
        LLK_ASSERT(tensor_shape.total_num_faces() == NUM_FACES, "FP32 block row max requires a 32x32 tile");
        LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "FP32 block row max requires full-height faces");
    }
    else
    {
        static_assert(block_ct_dim == 4, "16x32 block row max currently supports four tiles per block");
        LLK_ASSERT(tensor_shape.total_num_faces() == 2, "16-bit block row max currently requires 16x32 tiles");
        _llk_math_reduce_row_mop_config_<PoolType::MAX, ckernel::MathFidelity::LoFi>(tensor_shape);
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
 * Each tile has a distinct SrcB token. Releasing that token before the next
 * tile is required for reliable SrcB/MOVD2B bank selection.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    _set_dst_write_addr_by_rows_(dst_index);

    if constexpr (is_fp32_dest_acc_en)
    {
#pragma GCC unroll 4
        for (std::uint32_t tile = 0; tile < block_ct_dim; ++tile)
        {
            _llk_math_reduce_row_fp32_tile_();
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
        }
    }
    else
    {
#pragma GCC unroll 4
        for (std::uint32_t tile = 0; tile < block_ct_dim; ++tile)
        {
            ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
        }
    }
}
