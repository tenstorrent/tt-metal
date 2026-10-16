// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "lltt.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

/** @brief Record whole-tile row MAX pooling and its source-B result transpose. */
inline void reduce_row_max_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    const std::uint32_t replay_start  = ckernel::math::replay_buf_offset;
    const std::uint32_t num_faces     = tensor_shape.total_num_faces();
    const bool narrow_tile            = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;
    const std::uint32_t transpose_len = narrow_tile ? 11 : 13;
    const std::uint32_t replay_len    = num_faces + 1 + tensor_shape.num_faces_r_dim * transpose_len;

    lltt::record<lltt::NoExec>(replay_start, replay_len);
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        if (narrow_tile || face == 1)
        {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
    // Use only bank-zero modifiers; Wormhole move instructions have a two-bit address mode.
    for (std::uint32_t row = 0; row < tensor_shape.num_faces_r_dim; row++)
    {
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movd2b::MOV_1_ROW, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_ZEROSRC(0, 1, 0, 1);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        // SETRWC advances the destination carry counter, independently of ELWADD's counter.
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        if (!narrow_tile)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_B);
    }

    ckernel_template tmp(1, 1, lltt::replay_insn(replay_start, replay_len));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD));
    tmp.program();
}

/**
 * @brief Program a whole-tile reduce MOP, including fidelity phases and any result transpose.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining face count and destination stride.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity) && type != PoolType::MAX;
    constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
    const std::uint32_t replay_start    = ckernel::math::replay_buf_offset;
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        constexpr std::uint32_t replay_buf_len = 8;
        const std::uint32_t replay_offset      = replay_start + replay_buf_len - num_faces * 2;

        lltt::record<lltt::NoExec>(replay_start, replay_buf_len);
        // [0..7]: four faces, [4..7]: two faces, [6..7]: one face.
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);

        ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_offset, num_faces * 2));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
    else if constexpr (dim == ReduceDim::REDUCE_ROW && type == PoolType::MAX)
    {
        reduce_row_max_configure_mop(tensor_shape);
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        lltt::record<lltt::NoExec>(replay_start, 4);
        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
        }

        ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_start + 4 - num_faces, num_faces));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        constexpr std::uint32_t cross_pool_len = 4;
        constexpr std::uint32_t transpose_len  = 10;
        constexpr std::uint32_t final_pool_len = inner_loops;

        lltt::record<lltt::NoExec>(replay_start, cross_pool_len + transpose_len + final_pool_len);
        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 4);
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4);
        }
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
        TTI_GATESRCRST(0b1, 0b1);
        TTI_TRNSPSRCB;
        TTI_GATESRCRST(0b1, 0b1);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        TTI_GATESRCRST(0b1, 0b1);
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, 4);
        for (std::uint32_t fid = 1; fid < inner_loops; fid++)
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
        }
        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }

        ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_start + 4 - num_faces, num_faces));
        tmp.set_end_op(lltt::replay_insn(replay_start + cross_pool_len, transpose_len + final_pool_len));
        tmp.program();
    }
}

/**
 * @brief Reduce one tile with the MOP programmed by @ref _llk_math_reduce_init_.
 *
 * @tparam is_int_fpu_en: Must be false; the MOP has no int32-to-int8 cast.
 * @param dst_index: Tile index in the destination register.
 * @param tensor_shape: Tile geometry matching init.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape)
{
    static_assert(!is_int_fpu_en, "is_int_fpu_en is not supported by the Wormhole reduce: the MOP has no int32 to int8 cast");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_reduce_", tensor_shape);

    // Supported narrow tiles: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
}

/** @brief Program source, destination and fidelity address modifiers for a whole-tile reduce. */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity) && type != PoolType::MAX;

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        addr_mod_t {
            .srcb = {.incr = 8},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_0);

        if (tensor_shape.total_num_faces() == 4)
        {
            addr_mod_t {
                .srcb = {.incr = 24},
                .dest = {.incr = 24},
            }
                .set(ADDR_MOD_1);
            addr_mod_t {
                .srca = {.incr = 16},
                .srcb = {.incr = 16, .cr = 1},
                .dest = {.cr = 1},
            }
                .set(ADDR_MOD_2);
        }
        else if (tensor_shape.num_faces_c_dim == 2)
        {
            addr_mod_t {
                .srcb = {.incr = 8},
                .dest = {.cr = 1},
            }
                .set(ADDR_MOD_1);
        }
        else if (tensor_shape.num_faces_r_dim == 2)
        {
            addr_mod_t {
                .srcb = {.incr = 8},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_1);
        }

        addr_mod_t {
            .srca     = {.clr = 1, .cr = 1},
            .srcb     = {.clr = 1, .cr = 1},
            .dest     = {.clr = 1, .cr = 1},
            .fidelity = {.incr = high_fidelity ? 1u : 0u},
        }
            .set(ADDR_MOD_3);
    }
    else if constexpr (dim == ReduceDim::REDUCE_ROW && type == PoolType::MAX)
    {
        addr_mod_t {.srca = {.incr = 16}}.set(ADDR_MOD_0);
        if (tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim)
        {
            addr_mod_t {
                .srca = {.incr = 16},
                .dest = {.incr = 16},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 16},
                .dest = {.incr = 32},
            }
                .set(ADDR_MOD_1);
        }
        addr_mod_t {}.set(ADDR_MOD_2);
        addr_mod_t {
            .srcb = {.incr = 8},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_3);
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        if (tensor_shape.num_faces_c_dim > 1)
        {
            addr_mod_t {
                .srca = {.incr = 16},
                .dest = {.incr = 16},
            }
                .set(ADDR_MOD_0);
        }
        else
        {
            addr_mod_t {.srca = {.incr = 16}}.set(ADDR_MOD_0);
        }
        addr_mod_t {
            .srca = {.incr = 16},
            .dest = {.cr = 1},
        }
            .set(ADDR_MOD_1);
        addr_mod_t {
            .srca     = {.clr = 1, .cr = 1},
            .dest     = {.clr = 1, .cr = 1},
            .fidelity = {.incr = high_fidelity ? 1u : 0u},
        }
            .set(ADDR_MOD_2);
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        addr_mod_t {.fidelity = {.clr = 1}}.set(ADDR_MOD_0);
        addr_mod_t {.fidelity = {.incr = 1}}.set(ADDR_MOD_1);
        addr_mod_t {.srca = {.incr = 16}}.set(ADDR_MOD_2);
        addr_mod_t {
            .srca     = {.clr = 1, .cr = 1},
            .srcb     = {.clr = 1},
            .fidelity = {.incr = high_fidelity ? 1u : 0u},
        }
            .set(ADDR_MOD_3);
    }
}

/** @brief Initialize a whole-tile reduce; pair with @ref _llk_math_reduce_uninit_. */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    reduce_configure_addrmod<type, dim, math_fidelity>(tensor_shape);
    reduce_configure_mop<type, dim, math_fidelity>(tensor_shape);
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);

    if constexpr (type == PoolType::MAX && dim == ReduceDim::REDUCE_ROW)
    {
        // Preserve values with a zero low byte through the MOVD2B/ELWADD transpose.
        math::_configure_preserve_zero_flag_state_();
    }
    else
    {
        math::_configure_default_zero_flag_state_();
    }
}

/** @brief Restore the operand-driven zero-flag state after a reduce. */
inline void _llk_math_reduce_uninit_()
{
    math::_configure_default_zero_flag_state_();
}
