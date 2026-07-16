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
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Configure the reduce MOP for all paths: REDUCE_ROW SUM/AVG records a dual replay-buffer
 *        MOP, high-fidelity COL/SCALAR programs a multi-phase GAPOOL loop, all other paths need no MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining face count and dest stride.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity) && type != PoolType::MAX;

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        constexpr std::uint32_t replay_buf_len = 8;
        constexpr std::uint32_t inner_loops    = high_fidelity ? to_underlying(math_fidelity) : 1;
        const std::uint32_t replay_start       = ckernel::math::replay_buf_offset;
        const std::uint32_t num_faces          = tensor_shape.total_num_faces();
        const std::uint32_t replay_offset      = (num_faces == 4) ? replay_start : replay_start + 4;

        load_replay_buf(
            replay_start,
            replay_buf_len,
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
            });

        ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_offset, num_faces * 2));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
    else if constexpr (dim == ReduceDim::REDUCE_ROW && type == PoolType::MAX)
    {
        const std::uint32_t replay_start = ckernel::math::replay_buf_offset;
        const std::uint32_t num_faces    = tensor_shape.total_num_faces();

        constexpr std::uint32_t gmpool_buf_len    = 4;
        constexpr std::uint32_t transpose_buf_len = 6;

        const std::uint32_t gmpool_len    = num_faces;
        const std::uint32_t gmpool_offset = replay_start + (tensor_shape.num_faces_r_dim > 1 ? 0 : 4 - num_faces);

        const std::uint32_t transpose_offset = replay_start + gmpool_buf_len;
        const std::uint32_t transpose_len    = tensor_shape.num_faces_r_dim > 1 ? 12 : 6;

        load_replay_buf(
            replay_start,
            gmpool_buf_len + 2 * transpose_buf_len,
            []
            {
                // [0..3]: 4 faces, [0..1]: 2 row faces, [2..3]: 2 column faces
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
                // [4-9]: transpose face 0
                TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_4, p_movd2b::MOV_1_ROW, 0);
                TTI_TRNSPSRCB;
                TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
                TTI_ZEROSRC(0, 1, 0, 1);
                TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
                TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, 0);
                // [10-15]: transpose face 1
                TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_4, p_movd2b::MOV_1_ROW, 0);
                TTI_TRNSPSRCB;
                TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
                TTI_ZEROSRC(0, 1, 0, 1);
                TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
                TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, 0);
            });

        ckernel_template tmp(1, 1, lltt::replay_insn(gmpool_offset, gmpool_len), lltt::replay_insn(transpose_offset, transpose_len));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD));
        tmp.program();
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
        const std::uint32_t replay_start    = ckernel::math::replay_buf_offset;
        const std::uint32_t num_faces       = tensor_shape.total_num_faces();

        load_replay_buf(
            replay_start,
            4,
            []
            {
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
            });

        const std::uint32_t replay_offset = (num_faces == 4) ? replay_start : replay_start + 2;
        ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_offset, num_faces));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        tmp.program();
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        constexpr std::uint32_t inner_loops    = high_fidelity ? to_underlying(math_fidelity) : 1;
        constexpr std::uint32_t cross_pool_len = 4;
        constexpr std::uint32_t transpose_len  = 7;
        constexpr std::uint32_t final_pool_len = inner_loops;
        constexpr std::uint32_t buf_len        = cross_pool_len + transpose_len + final_pool_len;
        const std::uint32_t replay_start       = ckernel::math::replay_buf_offset;
        const std::uint32_t num_faces          = tensor_shape.total_num_faces();

        load_replay_buf(
            replay_start,
            buf_len,
            []
            {
                // [0-3]: initial pool into dest[4]
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
                // [4-10]: transpose pooled row to column in SrcA
                TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
                TTI_TRNSPSRCB;
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
                TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);
                // [11..14]: final pool with unrolled fidelity
                if constexpr (high_fidelity)
                {
                    for (std::uint32_t fid = 0; fid < inner_loops - 1; fid++)
                    {
                        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
                    }
                }
                if constexpr (type == PoolType::MAX)
                {
                    TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                }
                else
                {
                    TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                }
            });

        const std::uint32_t pool_offset = replay_start + (4 - num_faces);
        const std::uint32_t end_op_len  = transpose_len + final_pool_len;

        ckernel_template tmp(1, inner_loops, lltt::replay_insn(pool_offset, num_faces));
        tmp.set_end_op(lltt::replay_insn(replay_start + cross_pool_len, end_op_len));
        tmp.program();
    }
}

/**
 * @brief Perform a reduction on the math thread, pooling faces into the destination register.
 *
 * REDUCE_ROW SUM/AVG uses MVMUL to multiply each face row by a column-scaler in SrcA, producing the reduced
 * column directly in dest; MAX pools each face row then transposes the result into dest via SrcB.
 * REDUCE_COL pools down rows of faces; REDUCE_SCALAR pools all faces then transposes the partial result
 * into a single column for a final pool.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath (casts int32 dest datums to int8 before moving to SrcB).
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tensor shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this
 *       function, and @ref _llk_math_reduce_uninit_ after it to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
}

/**
 * @brief Program address mod registers for the reduce operation.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining dest stride for REDUCE_ROW.
 */
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
        else
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
        if (tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim)
        {
            addr_mod_t {
                .srca = {.incr = 16},
                .dest = {.incr = 16},
            }
                .set(ADDR_MOD_0);

            addr_mod_t {
                .srca = {.clr = 1, .cr = 1},
                .srcb = {.clr = 1},
                .dest = {.clr = 1, .cr = 1},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 16},
            }
                .set(ADDR_MOD_0);

            addr_mod_t {
                .srca = {.incr = 16},
                .dest = {.incr = 32},
            }
                .set(ADDR_MOD_1);
        }

        addr_mod_t {
            .srca = {.clr = 1, .cr = 1},
            .srcb = {.clr = 1},
            .dest = {.clr = 1, .cr = 1},
        }
            .set(ADDR_MOD_2);

        addr_mod_t {
            .srcb = {.incr = 8},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_3);

        addr_mod_t {}.set(ADDR_MOD_4);

        if (tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim)
        {
            addr_mod_t {
                .srcb = {.clr = 1},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_5);
        }
        else
        {
            addr_mod_t {
                .srcb = {.clr = 1},
                .dest = {.incr = 24},
            }
                .set(ADDR_MOD_5);
        }

        addr_mod_t {
            .srcb = {.incr = 16},
        }
            .set(ADDR_MOD_6);
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
            addr_mod_t {
                .srca = {.incr = 16},
            }
                .set(ADDR_MOD_0);
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
        addr_mod_t {
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_0);

        addr_mod_t {.fidelity = {.incr = 1}}.set(ADDR_MOD_1);

        addr_mod_t {
            .srca = {.incr = 16},
        }
            .set(ADDR_MOD_2);

        addr_mod_t {
            .srca     = {.clr = 1, .cr = 1},
            .srcb     = {.clr = 1},
            .fidelity = {.incr = high_fidelity ? 1u : 0u},
        }
            .set(ADDR_MOD_3);
    }
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation: programs address mods and the MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_ with matching template args after this function,
 *       and @ref _llk_math_reduce_uninit_ after the last reduce to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    reduce_configure_addrmod<type, dim, math_fidelity>(tensor_shape);
    reduce_configure_mop<type, dim, math_fidelity>(tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    if constexpr (type == PoolType::MAX && dim == ReduceDim::REDUCE_ROW)
    {
        // The MOVD2B/ELWADD below read the Src zero substitution flag (FlushDenormals = !flag).
        // A datum whose low byte is zero (e.g. bf16 0x4400 = 768.0) would be flushed to 0 mid-reduction,
        // corrupting the sum. Disable the flag (via the math state tracker) around the transpose+add, then
        // return it to the operand driven baseline. WH does the same in its fp32 transpose.
        math::_configure_mov_ops_zero_flag_state_();
    }
}

/**
 * @brief Uninitialize after a reduce operation, undoing any init/execute-time workarounds.
 *
 * @note Reverses @ref _llk_math_reduce_init_
 */
inline void _llk_math_reduce_uninit_()
{
    // Restore the operand-driven baseline for the currently configured formats.
    math::_configure_default_zero_flag_state_(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt);
}
