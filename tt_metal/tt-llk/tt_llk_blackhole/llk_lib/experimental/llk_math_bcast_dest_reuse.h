// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void bcast_dest_reuse_configure_mop(const std::uint32_t num_faces = 4, const std::uint32_t acc_to_dest = 0)
{
    static_assert(
        bcast_type == BroadcastType::SCALAR || bcast_type == BroadcastType::COL, "bcast_dest_reuse supports BroadcastType::SCALAR and BroadcastType::COL only");
    if constexpr (bcast_type == BroadcastType::COL)
    {
        LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for BroadcastType::COL");
    }
    else
    {
        LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    }

    constexpr bool high_fidelity  = is_high_fidelity(math_fidelity);
    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL) ? p_elwise::SRCB_BCAST_COL : p_elwise::SRCB_BCAST_ALL;

    // Broadcast preserves SrcB across all faces and tiles; CLR_B is issued after the MOP.
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
    {
        ckernel_template tmp(
            num_tiles,
            num_faces,
            TT_OP_ELWADD(0, acc_to_dest, broadcast_type, ADDR_MOD_0, 0),
            TT_OP_ELWADD(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWADD(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWADD(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
    {
        ckernel_template tmp(
            num_tiles,
            num_faces,
            TT_OP_ELWSUB(0, acc_to_dest, broadcast_type, ADDR_MOD_0, 0),
            TT_OP_ELWSUB(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            if (num_faces == 4)
            {
                // For full 4-face tiles, ADDR_MOD_3 advances cleanly across face and tile boundaries,
                // so a single MOP covers all num_tiles tiles without per-tile host launches.
                ckernel_template tmp(
                    num_tiles * 4,
                    to_underlying(math_fidelity),
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_0, 0),
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_2, 0));
                tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
                tmp.program();
            }
            else
            {
                ckernel_template tmp(
                    num_faces,
                    to_underlying(math_fidelity),
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_0, 0),
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_2, 0));
                tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_4, 0));
                tmp.program();
            }
        }
        else
        {
            ckernel_template tmp(
                num_tiles, num_faces, TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_0, 0), TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_2, 0));
            tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
            tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
            tmp.program();
        }
    }
}

template <BroadcastType bcast_type>
inline void bcast_dest_reuse_move_dest_to_srcb()
{
    static_assert(
        bcast_type == BroadcastType::SCALAR || bcast_type == BroadcastType::COL, "bcast_dest_reuse supports BroadcastType::SCALAR and BroadcastType::COL only");

    // Wait for prior SFPU/MATH writes to DEST and for the unpacker's dummy SRCB_VLD before
    // copying broadcast operands from DEST into SrcB.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::MATH | p_stall::SRCB_VLD);
    if constexpr (bcast_type == BroadcastType::COL)
    {
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
    }
    else
    {
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 0);
    }
}

/**
 * @brief Output[tile] = SrcA[tile] (+, -, *) bcast(DST[src_index]) for num_tiles tiles.
 *
 * Moves either element [0] (BroadcastType::SCALAR) or Face 0 column 0 across all 16 rows
 * (BroadcastType::COL) from DST[src_index] into SrcB via MOVD2B, then broadcasts it across
 * num_tiles consecutive SrcA tiles into DEST starting at dst_index.
 *
 * @tparam eltwise_binary_type FPU op: ELWADD, ELWSUB, or ELWMUL.
 * @tparam bcast_type Broadcast mode: BroadcastType::SCALAR or BroadcastType::COL.
 * @tparam num_tiles Number of consecutive tiles walked per call (matches the unpack init).
 * @tparam Dst Destination sync mode (DstSync::SyncHalf or DstSync::SyncFull).
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled.
 * @tparam math_fidelity FPU math fidelity (non-LoFi phases apply to ELWMUL).
 * @tparam clear_dest When true, ZEROACC the acquired DEST section between MOVD2B and the MOP.
 *         ELWMUL accumulates into DEST, so set true unless the output tiles are already zero
 *         (and no other tile in the section is live).
 * @param src_index DEST tile holding the broadcast scalar or Face 0 column-0 vector.
 * @param dst_index Starting DEST tile for the output tiles.
 * @param tensor_shape Tile geometry (face_r_dim == FACE_R_DIM and num_faces in {2, 4} for COL;
 *        num_faces in {1, 2, 4} for SCALAR).
 * @note No op that leaves the Src zero-substitution flag at keep (copy_init / datacopy) may run
 *       between @ref _llk_math_bcast_dest_reuse_init_ and this call: the MOP would keep denormal
 *       Src operands it should flush. Checked on entry under LLK asserts.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest = false>
inline void _llk_math_bcast_dest_reuse_(
    const std::uint32_t src_index, const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    static_assert(
        bcast_type == BroadcastType::SCALAR || bcast_type == BroadcastType::COL, "bcast_dest_reuse supports BroadcastType::SCALAR and BroadcastType::COL only");
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWADD || eltwise_binary_type == EltwiseBinaryType::ELWSUB ||
            eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "bcast_dest_reuse supports ELWADD, ELWSUB, and ELWMUL");

    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_bcast_dest_reuse_", tensor_shape);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if constexpr (bcast_type == BroadcastType::COL)
    {
        LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "bcast_dest_reuse with BroadcastType::COL requires 16-row faces");
        LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for BroadcastType::COL");
    }
    else
    {
        LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    }

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // The init re-established the operand-driven flag; an op issued between it and this call must not have
    // left keep behind, or the MOP keeps the denormal Src operands it should flush (fires only under LLK asserts).
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "bcast_dest_reuse: Src zero-substitution flag does not hold the operand-driven value; an op between the init "
        "and this call left it at keep");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(src_index);
    bcast_dest_reuse_move_dest_to_srcb<bcast_type>();
    if constexpr (clear_dest)
    {
        if constexpr (Dst == DstSync::SyncFull)
        {
            TT_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
        }
        else
        {
            static_assert(Dst == DstSync::SyncHalf);
            TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id);
        }
    }
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        ckernel_template::run();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            if (num_faces == 4)
            {
                ckernel_template::run();
            }
            else
            {
                for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++)
                {
                    ckernel_template::run();
                }
            }
        }
        else
        {
            ckernel_template::run();
        }
    }
    // Manually clear B once MOP is done for broadcast reuse.
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void bcast_dest_reuse_configure_addrmod(const std::uint32_t num_faces)
{
    static_assert(
        bcast_type == BroadcastType::SCALAR || bcast_type == BroadcastType::COL, "bcast_dest_reuse supports BroadcastType::SCALAR and BroadcastType::COL only");

    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;
    constexpr std::int16_t srcb_row_incr       = (bcast_type == BroadcastType::COL) ? 8 : 0;

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        addr_mod_t {
            .srca = {.incr = 8},
            .srcb = {.incr = srcb_row_incr},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_0);
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity)
        {
            addr_mod_t {
                .srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = fidelity_increment}}
                .set(ADDR_MOD_2);

            addr_mod_t {
                .srca     = {.incr = 0, .clr = 1},
                .srcb     = {.incr = 0, .clr = 1},
                .dest     = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_3);

            addr_mod_t {
                .srca     = {.incr = 0, .clr = 1},
                .srcb     = {.incr = 0, .clr = 1},
                .dest     = {.incr = static_cast<std::int16_t>(8 + (4 - num_faces) * 16), .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_4);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 0, .clr = 1},
                .srcb = {.incr = 0, .clr = 1},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_2);
            addr_mod_t {
                .srca = {.incr = 0, .clr = 1},
                .srcb = {.incr = 0, .clr = 1},
                .dest = {.incr = static_cast<std::int16_t>(8 + (4 - num_faces) * 16)},
            }
                .set(ADDR_MOD_3);
        }
    }
}

/**
 * @brief Initialize the FPU for @ref _llk_math_bcast_dest_reuse_: addr-mods and the num_tiles MOP.
 *
 * @tparam eltwise_binary_type FPU op: ELWADD, ELWSUB, or ELWMUL.
 * @tparam bcast_type Broadcast mode: BroadcastType::SCALAR or BroadcastType::COL.
 * @tparam num_tiles Number of consecutive tiles walked per call.
 * @tparam math_fidelity FPU math fidelity.
 * @param tensor_shape Tile geometry (face_r_dim == FACE_R_DIM and num_faces in {2, 4} for COL;
 *        num_faces in {1, 2, 4} for SCALAR).
 * @param acc_to_dest Dest-accumulate field of the ELWADD/ELWSUB MOP instructions.
 * @note Establishes the operand-driven default Src zero-substitution state, which the execute asserts under
 *       LLK asserts; an op that leaves it at keep (copy_init / datacopy) must be followed by this init again.
 *       Pairs with @ref _llk_unpack_A_bcast_dest_reuse_init_ on the unpack thread.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void _llk_math_bcast_dest_reuse_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t acc_to_dest = 0)
{
    static_assert(
        bcast_type == BroadcastType::SCALAR || bcast_type == BroadcastType::COL, "bcast_dest_reuse supports BroadcastType::SCALAR and BroadcastType::COL only");
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWADD || eltwise_binary_type == EltwiseBinaryType::ELWSUB ||
            eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "bcast_dest_reuse supports ELWADD, ELWSUB, and ELWMUL");

    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_bcast_dest_reuse_init_", tensor_shape);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if constexpr (bcast_type == BroadcastType::COL)
    {
        LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "bcast_dest_reuse with BroadcastType::COL requires 16-row faces");
        LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for BroadcastType::COL");
    }
    else
    {
        LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    }

    bcast_dest_reuse_configure_addrmod<eltwise_binary_type, bcast_type, math_fidelity>(num_faces);
    bcast_dest_reuse_configure_mop<eltwise_binary_type, bcast_type, num_tiles, math_fidelity>(num_faces, acc_to_dest);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    // ELWADD/ELWMUL/ELWSUB read the Src zero-substitution flag and the addr-mod/MOP setup above does not touch
    // it, so re-establish the operand-driven DEFAULT here -- otherwise a preceding copy_init/datacopy op that
    // left PRESERVE leaks into the MOP (denormal Src results differ). Mirrors _llk_math_eltwise_binary_init_.
    math::_configure_default_zero_flag_state_();
}
