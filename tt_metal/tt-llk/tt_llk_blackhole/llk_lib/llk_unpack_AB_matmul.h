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

using namespace ckernel;
using namespace ckernel::unpacker;

// RISC-side mirror of the unpacker state the matmul call programs through the instruction stream: the base address
// register of each unpacker in each config context (16-byte words) and the two scratch registers the CFGSHIFTMASK
// steps add. 0xFFFFFFFF marks an unknown value: _llk_unpack_AB_matmul_init_ resets every entry (another op may have
// written the registers in between), and the first call after the init writes the registers from scratch.
constexpr std::uint32_t MATMUL_UNP_UNKNOWN = 0xFFFFFFFF;
static std::uint32_t matmul_unp_base[2][2] = {{MATMUL_UNP_UNKNOWN, MATMUL_UNP_UNKNOWN}, {MATMUL_UNP_UNKNOWN, MATMUL_UNP_UNKNOWN}}; // [context][unpacker]
static std::uint32_t matmul_unp_scratch[2] = {MATMUL_UNP_UNKNOWN, MATMUL_UNP_UNKNOWN}; // [0] SCRATCH_SEC0 streamed tile stride, [1] SCRATCH_SEC1 held tile size

/**
 * @brief Set the SrcA address step used to stream matmul in1 columns.
 *
 * tile_size is in unpacker L1 address units (16-byte words), not bytes. Only the
 * column-streaming path (ct_dim >= rt_dim, no in1 kernel broadcast) uses this step.
 * A stride of one restores the normal contiguous-tile step.
 */
inline void _llk_unpack_AB_matmul_set_in1_column_stride_(const std::uint32_t tile_size, const std::uint32_t stride_tiles)
{
    LLK_ASSERT(tile_size > 0 && tile_size <= 0xffff, "Matmul tile size must fit the SrcA address-step register");
    LLK_ASSERT(stride_tiles > 0 && stride_tiles <= 0xffff / tile_size, "Matmul column stride must fit the SrcA address-step register");
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size * stride_tiles), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
}

/**
 * @brief Record the replay body for one streamed tile of a matmul row.
 *
 * The body is the UNPACR group that moves one tile into SRC (one full-tile UNPACR, or a zero-source NOP,
 * two face UNPACRs and a Z counter reset for a partial face), then the L1 base address advance of that
 * unpacker, then the NOP that covers the two cycles of the config write before the next UNPACR reads the
 * base address. The advance is a single CFGSHIFTMASK that adds SCRATCH_SEC0_val to CFG_REG (the base
 * address register of one config context); @ref _llk_unpack_AB_matmul_ keeps the scratch register loaded
 * with the streamed operand's tile stride. Under kernel broadcast the advance is a NOP so the same tile is
 * re-read.
 *
 * @tparam SRC: SrcA or SrcB, the source register (and unpacker) of the streamed operand.
 * @tparam CFG_REG: Base address register of the streamed operand's unpacker in the config context this copy serves.
 * @tparam ADVANCE: False under kernel broadcast (the base address is not advanced).
 * @param partial_face: Whether the streamed operand is unpacked face-by-face.
 */
template <std::uint32_t SRC, std::uint32_t CFG_REG, bool ADVANCE>
inline void _llk_unpack_AB_matmul_stream_tile_body_(const bool partial_face)
{
    if (partial_face)
    {
        TTI_UNPACR_NOP(SRC, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_SETADCZW((SRC == SrcA) ? p_setadc::UNP_A : p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
    }
    else
    {
        TTI_UNPACR(SRC, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
    }

    if constexpr (ADVANCE)
    {
        // CFG_REG = CFG_REG + SCRATCH_SEC0_val (operation 0b011 is add, the mask covers all 32 bits, scratch_sel 0 is
        // SCRATCH_SEC0). One two-cycle instruction instead of RDCFG, ADDDMAREG, STALLWAIT and WRCFG, whose wait
        // for the THCON add held the thread for about 16 cycles per streamed tile, twice the data time of an 8-bit tile.
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, CFG_REG);
    }
    else
    {
        TTI_NOP;
    }
    // The config write takes two cycles and the next UNPACR must see the new base address.
    TTI_NOP;
}

/**
 * @brief Program the unpacker MOP/replay buffer for a matmul operand unpack.
 *
 * Builds a replay buffer that unpacks the streamed (non-reused) operand and advances its L1 base
 * address by one tile each step. Which operand is reused versus streamed is chosen by comparing
 * ct_dim and rt_dim (reuse_a = ct_dim >= rt_dim); operand A maps to SrcB and operand B to SrcA.
 * The address advance is suppressed under kernel broadcast.
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

    const bool reuse_a = ct_dim >= rt_dim;
    // Two copies of the streamed tile body, one per config context: the UNPACR group (1 or 4 instructions), the
    // base address advance and the NOP.
    const std::uint32_t replay_buf_prog_len = (reuse_a && unpA_partial_face) ? 12 : ((!reuse_a && unpB_partial_face) ? 12 : 6);
    const std::uint32_t replay_buf_run_len  = replay_buf_prog_len / 2;

    if (reuse_a)
    {
        static_assert(kernel_broadcast_b <= 1, "kernel_broadcast>1 on matmul input 1 is not supported with reuse enabled");
        constexpr bool advance = (kernel_broadcast_b != 1);
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpA_partial_face]
            {
                _llk_unpack_AB_matmul_stream_tile_body_<SrcA, THCON_SEC0_REG3_Base_address_ADDR32, advance>(unpA_partial_face);
                _llk_unpack_AB_matmul_stream_tile_body_<SrcA, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, advance>(unpA_partial_face);
            });
    }
    else
    {
        static_assert(kernel_broadcast_a <= 1, "kernel_broadcast>1 on matmul input 0 is not supported with reuse enabled");
        constexpr bool advance = (kernel_broadcast_a != 1);
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpB_partial_face]
            {
                _llk_unpack_AB_matmul_stream_tile_body_<SrcB, THCON_SEC1_REG3_Base_address_ADDR32, advance>(unpB_partial_face);
                _llk_unpack_AB_matmul_stream_tile_body_<SrcB, THCON_SEC1_REG3_Base_cntx1_address_ADDR32, advance>(unpB_partial_face);
            });
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
 * face-by-face for partial faces), stashes kt_dim into a GPR for tile-size scaling, forgets the
 * unpacker register values the last matmul call left (the first call reprograms them), and programs
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
 * @note Call @ref _llk_unpack_AB_matmul_uninit_ to restore the modified datum-count state.
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
    // 16x16 matmul not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
    LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");

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

    // Another op may have written the base address and scratch registers since the last matmul call.
    for (std::uint32_t c = 0; c < 2; c++)
    {
        matmul_unp_base[c][0] = MATMUL_UNP_UNKNOWN;
        matmul_unp_base[c][1] = MATMUL_UNP_UNKNOWN;
        matmul_unp_scratch[c] = MATMUL_UNP_UNKNOWN;
    }

    _llk_unpack_AB_matmul_mop_config_<kernel_broadcast_a, kernel_broadcast_b>(ct_dim, rt_dim, unpA_partial_face, unpB_partial_face);
}

/**
 * @brief No-op after a matmul operation.
 *
 * x-start/x-end is transient and reprogrammed by the next operation's init (see tt-llk#1036), so
 * there is nothing to restore here.
 *
 * @note Call @ref _llk_unpack_AB_matmul_init_ before this function.
 */
inline void _llk_unpack_AB_matmul_uninit_()
{
}

/**
 * @brief Unpack one tile into SRC with the matmul's held-operand UNPACR group.
 *
 * @tparam SRC: SrcA or SrcB.
 * @param partial_face: Whether the operand is unpacked face-by-face.
 */
template <std::uint32_t SRC>
inline void _llk_unpack_AB_matmul_held_tile_(const bool partial_face)
{
    if (partial_face)
    {
        TTI_UNPACR_NOP(SRC, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        // Do face by face unpacking
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_SETADCZW((SRC == SrcA) ? p_setadc::UNP_A : p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
    }
    else
    {
        TTI_UNPACR(SRC, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
    }
}

/**
 * @brief Add a small signed multiple of a scratch register to a config register with CFGSHIFTMASK.
 *
 * One instruction per set bit of |m|: the scratch value is rotated left by the bit position (the rotation is a
 * multiplication while the value stays below 2^28, which a tile stride in 16-byte words does), operation 0b011 adds
 * and 0b111 subtracts it. |m| is at most 15.
 *
 * @tparam SCRATCH_SEL: 0 for SCRATCH_SEC0, 1 for SCRATCH_SEC1.
 * @tparam CFG_REG: The config register to move.
 * @param m: The signed multiple, -15 to 15, not 0.
 */
template <std::uint32_t SCRATCH_SEL, std::uint32_t CFG_REG>
inline void _llk_unpack_AB_matmul_step_reg_(const std::int32_t m)
{
    if (m > 0)
    {
        if (m & 1)
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, SCRATCH_SEL, CFG_REG);
        }
        if (m & 2)
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 31, SCRATCH_SEL, CFG_REG);
        }
        if (m & 4)
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 30, SCRATCH_SEL, CFG_REG);
        }
        if (m & 8)
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 29, SCRATCH_SEL, CFG_REG);
        }
    }
    else
    {
        const std::uint32_t a = static_cast<std::uint32_t>(-m);
        if (a & 1)
        {
            TTI_CFGSHIFTMASK(1, 0b111, 32 - 1, 0, SCRATCH_SEL, CFG_REG);
        }
        if (a & 2)
        {
            TTI_CFGSHIFTMASK(1, 0b111, 32 - 1, 31, SCRATCH_SEL, CFG_REG);
        }
        if (a & 4)
        {
            TTI_CFGSHIFTMASK(1, 0b111, 32 - 1, 30, SCRATCH_SEL, CFG_REG);
        }
        if (a & 8)
        {
            TTI_CFGSHIFTMASK(1, 0b111, 32 - 1, 29, SCRATCH_SEL, CFG_REG);
        }
    }
}

/**
 * @brief Bring one unpacker base address register to the address a row needs, through the instruction stream.
 *
 * When the register is known to hold `needed` nothing is issued. When it is known and the distance to `needed` is
 * `m_hint` times `unit` (the operand's tile stride, held in the scratch register SCRATCH_SEL), the register is moved
 * with @ref _llk_unpack_AB_matmul_step_reg_. Otherwise the address is written through the GPR with SETDMAREG and
 * WRCFG. Every write is a config instruction of this thread, so it lands after the UNPACRs of the previous row have
 * been accepted by the unpacker and before the UNPACRs of this row; no RISC-side register write and no context
 * semaphore round trip is involved.
 *
 * @tparam SCRATCH_SEL: Scratch register holding `unit`.
 * @tparam GPR: Scratch GPR for the full write.
 * @tparam CFG_REG: The base address register.
 * @param tracked: The RISC's mirror of the register (updated to `needed`).
 * @param needed: The address the row needs.
 * @param unit: The operand's tile stride in 16-byte words.
 * @param m_hint: The expected distance in units, -15 to 15 (0: no expectation).
 * @return Whether an instruction was issued (the caller pads the write before the next UNPACR).
 */
template <std::uint32_t SCRATCH_SEL, std::uint32_t GPR, std::uint32_t CFG_REG>
inline bool _llk_unpack_AB_matmul_set_base_(std::uint32_t &tracked, const std::uint32_t needed, const std::uint32_t unit, const std::int32_t m_hint)
{
    if (tracked == needed)
    {
        return false;
    }
    if (tracked != MATMUL_UNP_UNKNOWN && m_hint != 0 && (needed - tracked) == static_cast<std::uint32_t>(m_hint) * unit)
    {
        _llk_unpack_AB_matmul_step_reg_<SCRATCH_SEL, CFG_REG>(m_hint);
    }
    else
    {
        TT_SETDMAREG(0, LOWER_HALFWORD(needed), 0, LO_16(GPR));
        TT_SETDMAREG(0, UPPER_HALFWORD(needed), 0, HI_16(GPR));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(GPR, p_cfg::WRCFG_32b, CFG_REG);
    }
    tracked = needed;
    return true;
}

/**
 * @brief One row of a matmul call in config context CTX: program the two base addresses, unpack the held tile, stream the other operand.
 *
 * @tparam CTX: The config context (0 or 1) the call runs in.
 * @param reuse_a: Operand A (in0, SrcB, unpacker 1) is held and operand B (in1, SrcA, unpacker 0) streamed.
 * @param address_a: L1 address of the in0 tile of the row (16-byte words).
 * @param address_b: L1 address of the first in1 tile of the row.
 * @param stream_stride: Tile stride of the streamed operand (SCRATCH_SEC0).
 * @param held_unit: Tile size of the held operand (SCRATCH_SEC1).
 * @param m_stream: Expected distance of the streamed base from its tracked value, in strides.
 * @param m_held: Expected distance of the held base from its tracked value, in held tile sizes.
 * @param rut_dim: Streamed tiles of the row.
 * @param held_partial_face: Whether the held operand is unpacked face-by-face.
 */
template <std::uint32_t CTX>
inline void _llk_unpack_AB_matmul_row_(
    const bool reuse_a,
    const std::uint32_t address_a,
    const std::uint32_t address_b,
    const std::uint32_t stream_stride,
    const std::uint32_t held_unit,
    const std::int32_t m_stream,
    const std::int32_t m_held,
    const std::uint32_t rut_dim,
    const bool held_partial_face)
{
    // unpacker 0 (SrcA) reads in1 at address_b, unpacker 1 (SrcB) reads in0 at address_a
    constexpr std::uint32_t REG_UNP0 = (CTX == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    constexpr std::uint32_t REG_UNP1 = (CTX == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;

    bool written;
    if (reuse_a)
    {
        written = _llk_unpack_AB_matmul_set_base_<0, p_gpr_unpack::TMP0, REG_UNP0>(matmul_unp_base[CTX][0], address_b, stream_stride, m_stream);
        written |= _llk_unpack_AB_matmul_set_base_<1, p_gpr_unpack::TMP1, REG_UNP1>(matmul_unp_base[CTX][1], address_a, held_unit, m_held);
        // the replay advances the streamed base by one stride per tile
        matmul_unp_base[CTX][0] += rut_dim * stream_stride;
    }
    else
    {
        written = _llk_unpack_AB_matmul_set_base_<0, p_gpr_unpack::TMP1, REG_UNP1>(matmul_unp_base[CTX][1], address_a, stream_stride, m_stream);
        written |= _llk_unpack_AB_matmul_set_base_<1, p_gpr_unpack::TMP0, REG_UNP0>(matmul_unp_base[CTX][0], address_b, held_unit, m_held);
        matmul_unp_base[CTX][1] += rut_dim * stream_stride;
    }
    if (written)
    {
        // The config write takes two cycles; the UNPACR must see the new base address.
        TTI_NOP;
    }

    if (reuse_a)
    {
        _llk_unpack_AB_matmul_held_tile_<SrcB>(held_partial_face);
    }
    else
    {
        _llk_unpack_AB_matmul_held_tile_<SrcA>(held_partial_face);
    }

    // Stream the other operand; a set zmask bit selects the replay copy of context 1. The mask covers the 16
    // iterations a full-sync block can have.
    if (rut_dim == 1)
    {
        TTI_MOP(0, 0, (CTX == 0) ? 0 : 0xffff);
    }
    else
    {
        TT_MOP(0, rut_dim - 1, (CTX == 0) ? 0 : 0xffff);
    }
}

/**
 * @brief Unpack the operand tiles for a matmul (A x B) into SrcA and SrcB.
 *
 * Iterates over the reused dimension, computing per-tile L1 addresses (with optional kernel-
 * broadcast wraparound and kt_dim striding), and unpacks operand A to SrcB / operand B to SrcA
 * for each row while streaming the other operand through the MOP.
 *
 * The call runs in the current config context and switches to the other one at the end, as every unpack
 * operation does; it posts the UNPACK_SYNC token before its first instruction and takes it back with the SEMGET
 * after its last, so the next operation's RISC-side address writes stay off this context until the call's
 * UNPACRs have been accepted. Unlike the other unpack operations it writes no unpacker register from the RISC
 * and therefore does not read the semaphore before a row: the base addresses are moved through the instruction
 * stream (a CFGSHIFTMASK step per tile stride when the row is a small number of strides away from where the
 * register was left, which is the case in the k loops of matmul_tiles and matmul_block, a SETDMAREG and WRCFG
 * write otherwise), and the streamed tile stride (SCRATCH_SEC0) and the held tile size (SCRATCH_SEC1) are copied
 * from the tile size GPRs when they change, so a data format reconfig between calls is honoured without a
 * re-init. The RISC can queue at most 16 instructions ahead of the thread, so at most a handful of tokens are
 * outstanding and the semaphore never saturates.
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
    // In0/InA -> srcB (unpacker 1; supports partial face)
    // In1/InB -> srcA (unpacker 0)

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim; // rows of the block, one held tile each
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // streamed tiles per row

    // Tile strides in 16-byte words: the streamed operand advances one stride per tile (in1 tiles are consecutive,
    // in0 rows are kt_dim tiles apart), the held operand one tile size per row (in0) or per column (in1).
    const std::uint32_t stream_stride = reuse_a ? tile_size_b : (tile_size_a * kt_dim);
    const std::uint32_t held_unit     = reuse_a ? tile_size_a : tile_size_b;

    // Take the context token for this call (see the function description).
    semaphore_post(semaphore::UNPACK_SYNC); // Trisc::SEMPOST for context acquire

    // Hold the UNPACRs until the RISC-side post has landed; the RISC register path is not ordered against the
    // instruction stream and the SEMGET at the end of the call must not overtake the post.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Scratch registers from the tile size GPRs (TILE_SIZE_A is the in1 tile size, TILE_SIZE_B the in0 tile size)
    // when the strides differ from the ones programmed. Config instructions of this thread: ordered after the
    // CFGSHIFTMASKs of the previous call and before the ones of this call.
    bool scratch_written = false;
    if (matmul_unp_scratch[0] != stream_stride)
    {
        if (reuse_a)
        {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TILE_SIZE_A, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        }
        else
        {
            TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        }
        matmul_unp_scratch[0] = stream_stride;
        scratch_written       = true;
    }
    if (matmul_unp_scratch[1] != held_unit)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        if (reuse_a)
        {
            TTI_WRCFG(p_gpr_unpack::TILE_SIZE_B, p_cfg::WRCFG_32b, SCRATCH_SEC1_val_ADDR32);
        }
        else
        {
            TTI_WRCFG(p_gpr_unpack::TILE_SIZE_A, p_cfg::WRCFG_32b, SCRATCH_SEC1_val_ADDR32);
        }
        matmul_unp_scratch[1] = held_unit;
        scratch_written       = true;
    }
    if (scratch_written)
    {
        // The config write takes two cycles; a CFGSHIFTMASK may read the scratch register next.
        TTI_NOP;
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t offset_address_a = tile_size_a * (tile_index_a + (reuse_a ? (t * kt_dim) : (0)));
        std::uint32_t offset_address_b = tile_size_b * (tile_index_b + (reuse_a ? (0) : (t)));
        if constexpr (kernel_broadcast_a > 0)
        {
            offset_address_a = tile_size_a * ((tile_index_a + (reuse_a ? ((t * kt_dim)) : (0))) % kernel_broadcast_a);
        }
        if constexpr (kernel_broadcast_b > 0)
        {
            offset_address_b = tile_size_b * ((tile_index_b + (reuse_a ? (0) : (t))) % kernel_broadcast_b);
        }

        const std::uint32_t address_a = base_address_a + offset_address_a;
        const std::uint32_t address_b = base_address_b + offset_address_b;

        LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
        LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

        // Where the registers are expected to be, in tile strides. Row 0: the same context ran two calls ago; in a
        // k loop both operands have moved on by two tiles since, and the replay left the streamed base rut_dim
        // strides past that call's row start (in0 streamed: one in0 tile is one stride only when kt_dim is 1). Rows
        // after the first: the streamed base rewinds to the row start, the held base moves one row (kt_dim in0 tiles,
        // or one in1 tile).
        std::int32_t m_stream, m_held;
        if (t == 0)
        {
            m_stream = (reuse_a || kt_dim == 1) ? (2 - static_cast<std::int32_t>(rut_dim)) : 0;
            m_held   = 2;
        }
        else
        {
            m_stream = -static_cast<std::int32_t>(rut_dim);
            m_held   = reuse_a ? static_cast<std::int32_t>(kt_dim) : 1;
        }
        if (m_stream < -15 || m_stream > 15)
        {
            m_stream = 0;
        }
        if (m_held > 15)
        {
            m_held = 0;
        }

        const bool held_partial_face = reuse_a ? unpB_partial_face : unpA_partial_face;
        if (unp_cfg_context == 0)
        {
            _llk_unpack_AB_matmul_row_<0>(reuse_a, address_a, address_b, stream_stride, held_unit, m_stream, m_held, rut_dim, held_partial_face);
        }
        else
        {
            _llk_unpack_AB_matmul_row_<1>(reuse_a, address_a, address_b, stream_stride, held_unit, m_stream, m_held, rut_dim, held_partial_face);
        }
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
