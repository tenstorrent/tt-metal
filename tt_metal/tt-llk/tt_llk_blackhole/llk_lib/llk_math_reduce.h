// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

// local function declarations
template <PoolType type, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape);

template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop();

template <MathFidelity math_fidelity>
inline void reduce_row_sum_configure_mop(const ckernel::TensorShape& tensor_shape);

/**
 * @brief Transpose the pooled row result through SrcB so a row-reduction lands as a column in the destination register.
 *
 * Only used for MAX pool (GMPOOL does column-wise max of SrcA only, cannot use the operand-swap path).
 *
 * @tparam is_int_fpu_en: Cast int32 dest datums to int8 (via SFPU) before moving to SrcB.
 */
template <bool is_int_fpu_en>
inline void reduce_row_perform_transpose()
{
    if constexpr (is_int_fpu_en)
    {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 0);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 2);
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
        TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x1);
    }

    // Move pooled result to SrcB rows 16–31 and transpose
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);

    if constexpr (is_int_fpu_en)
    {
        TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x0);
    }

    // Add transposed SrcB into Dest through a zeroed SrcA
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_ZEROSRC(0, 1, 0, 1);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
}

/**
 * @brief Pool one face into the destination register, dispatching to the right pool instruction for the op type.
 *
 * MAX uses GMPOOL; SUM/AVG use GAPOOL directly (LoFi) or the preconfigured MOP followed by a DVALID clear (high fidelity).
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @tparam clear_mode: Source-clear mode applied after pooling (p_setrwc::CLR_* value).
 * @tparam index: Destination-register offset (the GMPOOL/GAPOOL dst field) the pooled result is written to.
 */
template <PoolType type, bool high_fidelity, std::uint32_t clear_mode, std::uint32_t index = 0>
inline void reduce_pool_op()
{
    // Pool face from SrcA (REDUCE_ROW: scaler; COL/SCALAR: data)
    if constexpr (type == PoolType::MAX)
    {
        TTI_GMPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
    else if constexpr (high_fidelity)
    {
        ckernel_template::run();
        if constexpr (clear_mode != p_setrwc::CLR_NONE)
        {
            TTI_CLEARDVALID(clear_mode, 0);
        }
    }
    else
    {
        TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
}

/**
 * @brief Pool one row of faces, clearing SrcA/B between faces and keeping the final result.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @param num_faces_c_dim: Number of column-faces in the row.
 */
template <PoolType type, bool high_fidelity>
inline void reduce_row_pool_all_faces(const std::uint32_t num_faces_c_dim)
{
    for (std::uint32_t col_num = 0; col_num < num_faces_c_dim - 1; col_num++)
    {
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
    }
    reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
}

inline void reduce_row_advance_dest(const bool is_narrow_tile)
{
    if (!is_narrow_tile)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Record the reduce-row SUM/AVG MVMUL sequence into the replay buffer and program the MOP.
 *
 * Records one face row (num_faces_c faces) of MVMULs. The MOP outer loop iterates over face rows.
 * Dest advances between face rows via ADDR_MOD_6's dest.cr mechanism.
 *
 * @tparam math_fidelity: Controls the number of fidelity phases per MVMUL half.
 * @param tensor_shape: Tile shape determining face count and dest row stride.
 */
template <MathFidelity math_fidelity>
inline void reduce_row_sum_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    constexpr std::uint32_t fid_count  = (math_fidelity == MathFidelity::LoFi) ? 1 : to_underlying(math_fidelity);
    const std::uint32_t replay_buf_len = tensor_shape.num_faces_c_dim * 2 * fid_count;

    auto record_replay = [&tensor_shape](const std::uint32_t last_face_addr_mod)
    {
        for (std::uint32_t faces_remaining = tensor_shape.num_faces_c_dim; faces_remaining > 0; faces_remaining--)
        {
            for (std::uint32_t p = 0; p < fid_count - 1; p++)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
            }
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);

            for (std::uint32_t p = 0; p < fid_count - 1; p++)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 8);
            }
            if (faces_remaining == 1)
            {
                TTI_MVMUL(p_setrwc::CLR_AB, 0, last_face_addr_mod, 8);
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_5, 8);
            }
        }
    };

    const std::uint32_t replay_start      = ckernel::math::replay_buf_offset;
    const std::uint32_t replay_last_start = replay_start + replay_buf_len;

    load_replay_buf(
        replay_start,
        replay_buf_len * 2,
        [&record_replay]
        {
            record_replay(ADDR_MOD_6);
            record_replay(ADDR_MOD_7);
        });

    const std::uint32_t replay_op      = lltt::replay_insn(replay_start, replay_buf_len);
    const std::uint32_t replay_last_op = lltt::replay_insn(replay_last_start, replay_buf_len);
    ckernel_template tmp(tensor_shape.num_faces_r_dim, 1, replay_op);
    tmp.set_last_outer_loop_instr(replay_last_op);
    tmp.program();
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        const bool is_narrow_tile = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;

        if constexpr (type == PoolType::MAX)
        {
            reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            reduce_row_perform_transpose<is_int_fpu_en>();

            if (tensor_shape.num_faces_r_dim > 1)
            {
                reduce_row_advance_dest(is_narrow_tile);
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
                reduce_row_perform_transpose<is_int_fpu_en>();
            }
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
        }
        else
        {
            ckernel_template::run();
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        for (std::uint32_t row_num = 0; row_num < tensor_shape.num_faces_r_dim; row_num++)
        {
            // Just pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            if (tensor_shape.num_faces_c_dim > 1)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

                reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            }
            // Reset Dest Counter
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
        TTI_GATESRCRST(0b1, 0b1);
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1, 0b1);
        // copy over all 16 rows from B to A
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        // gate math instructions until src A has been updated by MOV instructions
        TTI_GATESRCRST(0b1, 0b1);
        // zero out scratch in dest
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);

        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            if constexpr (high_fidelity)
            {
                for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
                {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
    }
}

template <PoolType type, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;

    addr_mod_t {
		.srca = {.incr = 0},
		.srcb = {.incr = 0},
		.dest = {.incr = 0},
		.fidelity = {.incr = 0, .clr = 1}
	}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca     = {.incr = 0},
        .srcb     = {.incr = 1},
        .dest     = {.incr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

    if constexpr (high_fidelity)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
            .fidelity = {.incr = fidelity_increment},
        }
            .set(ADDR_MOD_3);
    }

    if constexpr (type != PoolType::MAX)
    {
        addr_mod_t {
            .srcb     = {.incr = 8},
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_4);

        addr_mod_t {
            .srcb     = {.cr = 1},
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_5);

        if (tensor_shape.num_faces_c_dim == 2)
        {
            addr_mod_t {
                .srcb     = {.cr = 1},
                .dest     = {.incr = 32},
                .fidelity = {.clr = 1},
            }
                .set(ADDR_MOD_6);
        }
        else
        {
            addr_mod_t {
                .srcb     = {.cr = 1},
                .dest     = {.incr = 16},
                .fidelity = {.clr = 1},
            }
                .set(ADDR_MOD_6);
        }

        addr_mod_t {
            .srcb     = {.cr = 1},
            .dest     = {.cr = 1},
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_7);
    }
}

/**
 * @brief Build the high-fidelity reduce MOP: a multi-phase GAPOOL sequence (one inner-loop iteration per fidelity phase).
 *
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision; sets the inner-loop length, values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop()
{
    constexpr int inner_loop_len = to_underlying(math_fidelity); // inner loop length is the number of fidelity phases 0, 2, 3, 4 (LoFi, Hifi2, Hifi3, Hifi4)

    if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.program();
    }
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation: programs address mods and, for high fidelity, the pool MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    reduce_configure_addrmod<type, math_fidelity>(tensor_shape);

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        reduce_row_sum_configure_mop<math_fidelity>(tensor_shape);
    }
    else if constexpr (high_fidelity)
    {
        reduce_configure_mop<dim, math_fidelity>();
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Uninitialize after a reduce operation, undoing any init/execute-time workarounds.
 *
 * @note Reverses @ref _llk_math_reduce_init_
 */
inline void _llk_math_reduce_uninit_()
{
}
