// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

// local function declarations
template <bool is_32bit>
inline void transpose_dest_configure_addrmod();
template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop();

/**
 * @brief Transpose a full 32-bit tile in place at dest index 0 (relative to the current bank) via format switching.
 *
 * Because the FPU transposes 16-bit datums, each 32-bit datum is transposed as two halves: the high 16 bits are
 * transposed and written first, then the cached low 16 bits, so the low bits don't clobber the high bits.
 * transpose_of_faces additionally transposes the face arrangement, not just elements within each face.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @note Toggles DISABLE_IMPLIED_SRCA_FMT and the SrcA format/FP32/zero-flag config bits, restoring them on exit.
 */
template <bool transpose_of_faces>
inline void transpose_dest_32b()
{
    // Disable implied SrcA format inference so manual SrcA format switches take effect.
    // On Blackhole the ALU format is normally inferred; we must disable this for
    // the SrcA format switching approach to control MOVD2B/MOVB2D behavior.
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);

    // The 32b hi16/lo16 MOV sequence must not flush datums with a zero low byte; own the Src
    // zero-substitution flag via the math state tracker. Asserted here (execute) so it survives any
    // llk_math_hw_configure that ran after the init; skip-if-set keeps it cheap.
    math::ZeroFlags::execute_reconfig_mov_ops();

    // Transpose all the low 16 bit elements of all faces and put them in SrcA.
    // Eventually Dest=0, SrcA=0.
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    ckernel_template::run();

    if constexpr (!transpose_of_faces)
    {
        // Transpose Face 0 hi bits and put back in Dest register.
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // Dest=16, SrcA=0

        // Transpose Face 1 hi bits and put back in Dest register.
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // Dest=32, SrcA=0

        // Transpose Face 2 hi bits and put back in Dest register.
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // Dest=48, SrcA=0

        // Transpose Face 3 hi bits and put back in Dest register.
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 12); // Dest=0, SrcA=0

        // Since all high bits have been written. We can set the appropriate formats and
        // write all the transposed low bits to Dest register.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

        TTI_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 8);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 16);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 24);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 32, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 32);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 40, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 40);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 48, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 48);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 56, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 56);
    }
    else
    {
        // Transpose Face 0 hi bits and put back in Dest register.
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 12); // Dest=0, SrcA=0

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

        // Write transposed Face 0 low bits to Dest, to make room for Face 1 high bits
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 8); // Dest=16, SrcA=0

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);

        // Transpose Face 1 hi bits and move to SrcA
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
        lltt::replay(ckernel::math::replay_buf_offset + 8, 5);
        lltt::replay(ckernel::math::replay_buf_offset + 5, 3);
        TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // Dest=32, SrcA=0

        // Transpose Face 2 hi bits and move to Face 1's area in Dest
        lltt::replay(ckernel::math::replay_buf_offset + 8, 5);
        TTI_MOVB2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, (0x3ff & (-16 + 0)));
        TTI_MOVB2D(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, (0x3ff & (-16 + 4)));
        TTI_MOVB2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, (0x3ff & (-16 + 8)));
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, (0x3ff & (-16 + 12))); // Dest=32, SrcA=0

        // Write transposed Face 1 hi bits and move to Face 2's area in Dest
        TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
        TTI_MOVA2D(p_mov::DEST_NORM, 8, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 8); // Dest=48, SrcA=0

        // Transpose Face 3 hi bits and put back in Dest register
        lltt::replay(ckernel::math::replay_buf_offset + 8, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 12);

        // Since all high bits have been written. We can set the appropriate formats and
        // write all the low bits to Dest register.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

        // Write transposed Face 1 low bits to Face 2's area in Dest
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 32);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 40);

        // Write transposed Face 2 low bits to Face 1's area in Dest
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 32, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 16);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 40, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 24);

        // Write transposed Face 3 low bits to Face 3's area in Dest
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 48, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 48);
        TTI_MOVA2D(p_mov::DEST_32B_LOW, 56, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 56);
    }

    // Restore config state
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// Notes on these template parameters:
// 1. <transpose_of_faces=false, is_32bit=false>: not supported.
// 2. <transpose_of_faces=false, is_32bit=true>: 4x 16x16 face transpose; can be combined with _llk_unpack_A_ with transpose_of_faces=true.
// 3. <transpose_of_faces=true, is_32bit=false>: the default case (full 32x32 tile transpose, non-32-bit).
// 4. <transpose_of_faces=true, is_32bit=true>: full 32x32 tile transpose for 32-bit.
//
// We may want to revisit these template parameters, and perhaps the
// transpose_dest API generally as it's not currently widely used:
// https://github.com/tenstorrent/tt-llk/issues/290
/**
 * @brief Transpose a tile in place within the destination register.
 *
 * For 32-bit datums uses the format-switching path (low/high 16-bit halves transposed separately to avoid
 * clobbering); otherwise runs the preconfigured transpose MOP. transpose_of_faces controls whether the face
 * arrangement is transposed in addition to the elements within each face.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (uses the format-switching transpose path).
 * @param dst_index: Tile index into the destination register.
 * @note Call @ref _llk_math_transpose_dest_init_ with matching template args before this
 *       function, and @ref _llk_math_transpose_dest_uninit_ after it to restore modified state.
 * @note On the unpack thread, the tile must already be in dest (via @ref _llk_unpack_A_ datacopy);
 *       @ref _llk_unpack_set_srcb_dummy_valid_ marks SrcB valid so the MOVB2D/MOVD2B sequence can run.
 * @note <transpose_of_faces=false, is_32bit=false> is not supported.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait condition SRCA_VLD is required as MOVB2A doesn't automatically wait
    // for SrcA[MatrixUnit.SrcABank].AllowedClient == SrcClient::MatrixUnit.
    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);

    if constexpr (is_32bit)
    {
        transpose_dest_32b<transpose_of_faces>();
    }
    else
    {
        ckernel_unpack_template::run(2, 2);
    }

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

/**
 * @brief Program the address-mod slots for a destination-register transpose.
 *
 * @tparam is_32bit: True for 32-bit datums (uses SrcA-stepping mods for the format-switching path).
 */
template <bool is_32bit>
inline void transpose_dest_configure_addrmod()
{
    if constexpr (is_32bit)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 16},
        }
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t {
            .srca = {.incr = 16},
            .srcb = {.incr = 0},
            .dest = {.incr = 16},
        }
            .set(ADDR_MOD_2);

        addr_mod_t {
            .srca = {.clr = 1},
            .dest = {.clr = 1},
        }
            .set(ADDR_MOD_3);
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 16},
        }
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0x3ff & -16},
        }
            .set(ADDR_MOD_2);

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_3);
    }
}

/**
 * @brief Build the transpose MOP, recording the per-face MOVD2B/TRNSPSRCB/MOVB2D-MOVB2A sequence into the replay buffer.
 *
 * The 32-bit path records a 16-instruction replay buffer and drives it with a ckernel_template; the non-32-bit path
 * records a 15-instruction replay buffer and assembles a ckernel_unpack_template that runs the seven per-face steps.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (records the format-switching transpose sequence).
 */
template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop()
{
    if constexpr (is_32bit)
    {
        lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 16);

        TTI_MOVD2B(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(p_mov::DEST_32B_LOW, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(p_mov::DEST_32B_LOW, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(p_mov::DEST_32B_LOW, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);

        TTI_TRNSPSRCB;

        TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
        TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
        TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);

        TTI_MOVD2B(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(p_mov::DEST_NORM, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);

        TTI_TRNSPSRCB;

        TTI_MOVB2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
        TTI_MOVB2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);

        std::uint32_t loop_op_0          = lltt::replay_insn(ckernel::math::replay_buf_offset, 8);
        std::uint32_t loop_op_1          = TT_OP_MOVB2A(12, ADDR_MOD_2, p_movb2a::MOV_4_ROWS, 28);
        std::uint32_t loop_op_last_outer = TT_OP_MOVB2A(12, ADDR_MOD_3, p_movb2a::MOV_4_ROWS, 28);

        ckernel_template tmp(4, 1, loop_op_0, loop_op_1);
        tmp.set_last_outer_loop_instr(loop_op_last_outer);
        tmp.program();
    }
    else
    {
        load_replay_buf(
            16,
            15,
            []
            {
                // ABCD
                TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
                TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
                TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
                TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // dst += 16

                // EFGHIJKLM
                TTI_MOVD2B(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
                TTI_MOVD2B(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
                TTI_MOVD2B(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
                TTI_MOVD2B(p_mov::DEST_NORM, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
                TTI_TRNSPSRCB;
                TTI_MOVB2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
                TTI_MOVB2D(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
                TTI_MOVB2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
                TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // dst += 16

                // NO
                // Note: the 0x3ff mask ensures the negative offsets are 10 bits.
                TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (0 - 32)); // dst -= 16
                TTI_MOVA2D(p_mov::DEST_NORM, 8, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (8 - 16)); // dst -= 16
            });

        // The next 7 instructions expand as follows:
        // Face 0: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM)
        // Face 1: 4x MOVD2B, TRNSPSRCB, 4x MOVB2A, dst += 16 (EFGHI, ABCD..)
        // Face 2: 4x MOVD2B (dst -= 16), TRNSPSRCB, 4x MOVB2D, dst += 32 (..EFG, P, IJKL, Q)
        // Face 3: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM..)
        // Face 1: 2x MOVA2D (2x dst -= 16) (..NO)

        std::uint32_t EFGHIJKLM   = lltt::replay_insn(20, 9);
        std::uint32_t EFGHI       = lltt::replay_insn(20, 5);
        std::uint32_t ABCDEFG     = lltt::replay_insn(16, 7);
        std::uint32_t P           = TT_OP_MOVD2B(p_mov::DEST_NORM, 28, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 12); // dst -= 16
        std::uint32_t IJKL        = lltt::replay_insn(24, 4);
        std::uint32_t Q           = TT_OP_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 12); // dst += 32
        std::uint32_t EFGHIJKLMNO = lltt::replay_insn(20, 11);

        // The following MOP config simply runs the above 7 instructions in order (when executed with zmask 0b10):
        ckernel_unpack_template tmp(true, true, EFGHIJKLM, EFGHI, ABCDEFG, P, /* skip A */ Q, /* B */ IJKL, /* skip B */ EFGHIJKLMNO);
        tmp.program();
    }
}

/**
 * @brief Configure the math thread (address mods and MOP) for a destination-register transpose.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (configures the format-switching transpose path).
 * @note @ref _llk_math_transpose_dest_ runs the configured transpose with matching template args.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_init_()
{
    transpose_dest_configure_addrmod<is_32bit>();
    transpose_dest_configure_mop<transpose_of_faces, is_32bit>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
}

/**
 * @brief Uninitialize/cleanup after a destination-register transpose, restoring modified state to defaults.
 *
 * @note Reverses @ref _llk_math_transpose_dest_init_; currently a no-op since all state is transient.
 */
inline void _llk_math_transpose_dest_uninit_()
{
    // No state to restore - all states are transient or default
}
