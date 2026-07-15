// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

enum class GeneralizedMoeGateEltwiseBinaryMode
{
    COPY,
    RELOAD,
};

// local function declarations
template <EltwiseBinaryType eltwise_binary_type>
inline void generalized_moe_gate_eltwise_binary_configure_addrmod()
{
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        addr_mod_t {
            .srca = {.incr = 8},
            .srcb = {.incr = 8},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_0);
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);
    }
}

template <EltwiseBinaryType eltwise_binary_type, DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_generalized_moe_gate_eltwise_binary_(const std::uint32_t num_faces, std::uint32_t dst_index)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // bf16-only gate op (downstream topk static_asserts !is_fp32_dest_acc_en, and the idx|score 16-bit
    // packing needs bf16 DEST), so there is no fp32 dest accumulator to clear — unlike the standard
    // eltwise_binary, this op takes no clear_fp32_dst_acc / ZERO_ACC_MODE.
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    static_assert(!(eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity), "High fidelity is not supported for ELWMUL");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, GeneralizedMoeGateEltwiseBinaryMode mode, MathFidelity math_fidelity>
inline void generalized_moe_gate_eltwise_binary_configure_mop(const std::uint32_t acc_to_dest = 0, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity      = is_high_fidelity(math_fidelity);
    const std::uint32_t addr_mod      = ADDR_MOD_0;
    constexpr std::uint32_t innerloop = 16 >> 3; // 8 rows per eltwise op at a time.
    std::uint32_t outerloop           = num_faces;
    const auto broadcast_type         = p_elwise::SRCB_NO_BCAST;

    constexpr std::uint32_t dst_tile_offset = 64; // 1 tile x 64 rows per tile
    constexpr std::uint32_t dst_math_offset = 2 * dst_tile_offset;

    std::uint32_t math_op;

    // TODO: Probably not worth it to use a replay buffer/mop for this
    // Just hardcode the math ops in _llk_math_generalized_moe_gate_eltwise_binary_ since we only have 1 face
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
    {
        math_op = TT_OP_ELWADD(0, acc_to_dest, broadcast_type, addr_mod, dst_math_offset);
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
    {
        math_op = TT_OP_ELWSUB(0, acc_to_dest, broadcast_type, addr_mod, dst_math_offset);
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        static_assert(!high_fidelity, "High fidelity is not supported for ELWMUL");
        math_op = TT_OP_ELWMUL(0, 0, broadcast_type, addr_mod, dst_math_offset);
    }
    if constexpr (mode == GeneralizedMoeGateEltwiseBinaryMode::RELOAD)
    {
        // Record (without executing) the 5 instructions that move a whole face from DEST to SRCA, then
        // replay them as the mop's start op. The count below must match the 5 instructions recorded here.
        lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 5);
        TTI_STALLWAIT(
            p_stall::STALL_MATH,
            p_stall::WAIT_SFPU | p_stall::SRCA_VLD); // MOVD2A for a whole face assumes unpacker will set a dummy
                                                     // data_valid, so we want to wait on that
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 4);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 8);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 12);
        std::uint32_t replay_instr = lltt::replay_insn(math::replay_buf_offset, 5);
        ckernel_template tmp(outerloop, innerloop, math_op);
        tmp.set_start_op(replay_instr);
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0), math_op);
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
}

template <EltwiseBinaryType eltwise_binary_type, GeneralizedMoeGateEltwiseBinaryMode mode, MathFidelity math_fidelity>
inline void _llk_math_generalized_moe_gate_eltwise_binary_init_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    generalized_moe_gate_eltwise_binary_configure_addrmod<eltwise_binary_type>();

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        generalized_moe_gate_eltwise_binary_configure_mop<eltwise_binary_type, mode, math_fidelity>(acc_to_dest, num_faces);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}
