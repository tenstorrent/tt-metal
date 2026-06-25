// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "lltt.h"
#include "sanitizer/api.h"
#include "sfpi.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Program the unpacker MOP/replay buffer for a matmul operand unpack.
 *
 * Builds a replay buffer that unpacks the streamed (non-reused) operand and advances its L1 base
 * address by one tile each step. Which operand is reused versus streamed is chosen by comparing
 * ct_dim and rt_dim (reuse_a = ct_dim >= rt_dim); operand A maps to SrcB and operand B to SrcA.
 * The address bump is suppressed under kernel broadcast.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_mop_config_(
    const std::uint32_t ct_dim, const std::uint32_t rt_dim, const bool unpA_partial_face, const bool unpB_partial_face)
{
    // in0/inA - loaded to SrcB
    // in1/inB - loaded to SrcA

    const bool reuse_a                      = ct_dim >= rt_dim;
    const std::uint32_t replay_buf_prog_len = (reuse_a && unpA_partial_face) ? 16 : ((!reuse_a && unpB_partial_face) ? 16 : 10);
    const std::uint32_t replay_buf_run_len  = replay_buf_prog_len / 2;

    if (reuse_a)
    {
        static_assert(kernel_broadcast_b <= 1, "kernel_broadcast>1 on matmul input 1 is not supported with reuse enabled!");
        lltt::record(0, replay_buf_prog_len);
        if (unpA_partial_face)
        {
            TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
        }
        else
        {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        }
        if constexpr (kernel_broadcast_b == 1)
        {
            TTI_NOP;
            TTI_NOP;
            TTI_NOP;
        }
        else
        {
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        }
        TTI_NOP;
        if (unpA_partial_face)
        {
            TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
        }
        else
        {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        }
        if constexpr (kernel_broadcast_b == 1)
        {
            TTI_NOP;
            TTI_NOP;
            TTI_NOP;
        }
        else
        {
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        }
        TTI_NOP;
    }
    else
    {
        static_assert(kernel_broadcast_a <= 1, "kernel_broadcast>1 on matmul input 0 is not supported with reuse enabled!");
        lltt::record(0, replay_buf_prog_len);
        if (unpB_partial_face)
        {
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
        }
        else
        {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        }
        if constexpr (kernel_broadcast_a == 1)
        {
            TTI_NOP;
            TTI_NOP;
            TTI_NOP;
        }
        else
        {
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        }
        TTI_NOP;
        if (unpB_partial_face)
        {
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
        }
        else
        {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        }
        if constexpr (kernel_broadcast_a == 1)
        {
            TTI_NOP;
            TTI_NOP;
            TTI_NOP;
        }
        else
        {
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        }
        TTI_NOP;
    }

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                    // src B
        false,                                    // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_run_len), // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_run_len, replay_buf_run_len), // runs when context is 1
        0,
        0);

    tmp.program();
}

/**
 * @brief Initialize the unpacker for a matmul (A x B) operation.
 *
 * Re-enables within-face transpose if needed, programs per-unpacker datum counts (full-tile or
 * face-by-face for partial faces), stashes kt_dim into a GPR for tile-size scaling, and programs
 * the matmul MOP.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param transpose: Nonzero to enable within-face (16x16) transpose for SrcA.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @note Call @ref _llk_unpack_AB_matmul_uninit_ after this function to restore the modified datum-count state.
 * @ref _llk_unpack_AB_matmul_ is the matching execute call.
 * @ref _llk_math_matmul_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
__attribute__((always_inline)) inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    // 16x16 inputs not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
    LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");

    llk::san::unpack_operand_check(
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces);
    llk::san::operation_init<llk::san::Operation::UnpackABMatmul>(
        kernel_broadcast_a, kernel_broadcast_b, ct_dim, rt_dim, kt_dim, unpA_partial_face, unpB_partial_face);

    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    if (unpA_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        config_unpacker_x_end<p_setadc::UNP_A>(unpA_face_r_dim);
    }
    else
    {
        const std::uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    }

    if (unpB_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);
    }
    else
    {
        // Do full tile unpacking. No need to program face dim
        // as address counter pointing to the face is not incremented
        const std::uint32_t unpB_x_end = unpB_num_faces * unpB_face_r_dim * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
    }

    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM)); // store kt_dim to gpr for scaling tile size

    _llk_unpack_AB_matmul_mop_config_<kernel_broadcast_a, kernel_broadcast_b>(ct_dim, rt_dim, unpA_partial_face, unpB_partial_face);
}

/**
 * @brief No-op after a matmul operation.
 *
 * x-start/x-end is transient and reprogrammed by each operation's init (see tt-llk#1036), so there
 * is nothing to restore here.
 *
 * @note Call @ref _llk_unpack_AB_matmul_init_ before this function.
 */
inline void _llk_unpack_AB_matmul_uninit_()
{
}

/**
 * @brief Unpack the operand tiles for a matmul (A x B) into SrcA and SrcB.
 *
 * Iterates over the reused dimension, computing per-tile L1 addresses (with optional kernel-
 * broadcast wraparound and kt_dim striding), and unpacks operand A to SrcB / operand B to SrcA
 * for each step while synchronizing through the unpack semaphore and config-context switching.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param base_address_b: L1 base address of operand B's tile buffer.
 * @param tile_index_a: Starting tile index into operand A.
 * @param tile_index_b: Starting tile index into operand B.
 * @param tile_size_a: Size of one operand A tile, used to compute per-tile offsets.
 * @param tile_size_b: Size of one operand B tile, used to compute per-tile offsets.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @note Call @ref _llk_unpack_AB_matmul_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_matmul_uninit_ after it to restore modified state.
 * @ref _llk_math_matmul_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const bool unpA_partial_face = false,
    const bool unpB_partial_face = false,
    std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim   = 1,
    const std::uint32_t kt_dim   = 1)
{
    llk::san::operation_check<llk::san::Operation::UnpackABMatmul>(
        kernel_broadcast_a, kernel_broadcast_b, ct_dim, rt_dim, kt_dim, unpA_partial_face, unpB_partial_face);

    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    volatile std::uint32_t *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    if (!reuse_a)
    {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t offset_address_a      = tile_size_a * (tile_index_a + (reuse_a ? (t * kt_dim) : (0)));
        std::uint32_t next_offset_address_a = tile_size_a * (tile_index_a + (reuse_a ? ((t + 1) * kt_dim) : (0)));
        if constexpr (kernel_broadcast_a > 0)
        {
            offset_address_a      = tile_size_a * ((tile_index_a + (reuse_a ? ((t * kt_dim)) : (0))) % kernel_broadcast_a);
            next_offset_address_a = tile_size_a * ((tile_index_a + (reuse_a ? (((t + 1) * kt_dim)) : (0))) % kernel_broadcast_a);
        }
        std::uint32_t offset_address_b      = tile_size_b * (tile_index_b + (reuse_a ? (0) : (t)));
        std::uint32_t next_offset_address_b = tile_size_b * (tile_index_b + (reuse_a ? (0) : (t + 1)));
        if constexpr (kernel_broadcast_b > 0)
        {
            offset_address_b      = tile_size_b * ((tile_index_b + (reuse_a ? (0) : (t))) % kernel_broadcast_b);
            next_offset_address_b = tile_size_b * ((tile_index_b + (reuse_a ? (0) : ((t + 1)))) % kernel_broadcast_b);
        }
        std::uint32_t address_a      = base_address_a + offset_address_a;
        std::uint32_t next_address_a = base_address_a + next_offset_address_a;
        std::uint32_t address_b      = base_address_b + offset_address_b;
        std::uint32_t next_address_b = base_address_b + next_offset_address_b;

        // Wait for free context
        wait_for_next_context(2);

        _llk_unpack_configure_addresses_(address_b, address_a, cfg);

        semaphore_post(semaphore::UNPACK_SYNC); // Trisc::SEMPOST for context acquire

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        if (reuse_a)
        {
            if (unpB_partial_face)
            {
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                // Do face by face unpacking
                TTI_UNPACR(
                    SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_UNPACR(
                    SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
            }
            else
            {
                TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            }
            if ((t + 1) < t_dim)
            {
                // Let's load one more tile into srcB
                TT_SETDMAREG(0, LOWER_HALFWORD(next_address_a), 0, LO_16(p_gpr_unpack::TMP0));
                TT_SETDMAREG(0, UPPER_HALFWORD(next_address_a), 0, HI_16(p_gpr_unpack::TMP0));
                if (0 == unp_cfg_context)
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                else
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                TTI_DMANOP;
                if (unpB_partial_face)
                {
                    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                    // Do face by face unpacking
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                t++;
            }
        }
        else
        {
            if (unpA_partial_face)
            {
                // Do face by face unpacking
                TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR(
                    SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_UNPACR(
                    SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
            }
            else
            {
                TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            }
            if ((t + 1) < t_dim)
            {
                // Let's load one more tile into srcA
                TT_SETDMAREG(0, LOWER_HALFWORD(next_address_b), 0, LO_16(p_gpr_unpack::TMP0));
                TT_SETDMAREG(0, UPPER_HALFWORD(next_address_b), 0, HI_16(p_gpr_unpack::TMP0));
                if (0 == unp_cfg_context)
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                else
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                TTI_DMANOP;
                if (unpA_partial_face)
                {
                    // Do face by face unpacking
                    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                t++;
            }
        }

        TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xff); // Run the MOP

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}
