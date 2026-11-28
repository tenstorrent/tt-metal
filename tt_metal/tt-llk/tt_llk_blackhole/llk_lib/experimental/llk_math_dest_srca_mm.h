// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_matmul.h"

using namespace ckernel;
using namespace ckernel::math;

// DEST_SRCA_MM: matmul whose in1 is a Float32 DEST tile (e.g. an SFPU-decoded weight) moved to SrcA by MOVD2A, with no
// round trip through L1; in0 comes from L1 into SrcB (_llk_unpack_dest_srca_mm_).

/**
 * @brief Configure the math thread for dest_srca_mm: the matmul's address mods and MOP, SrcA format TF32.
 *
 * @tparam math_fidelity: Math fidelity, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam in0_rows: Rows of in0 (SrcB) the matmul computes, values = <8/16/32>
 * @note Disables implied SrcA format; @ref _llk_math_dest_srca_mm_uninit_ restores it.
 * @note On the unpack thread, pair with @ref _llk_unpack_dest_srca_mm_init_.
 */
template <MathFidelity math_fidelity, std::uint32_t in0_rows>
inline void _llk_math_dest_srca_mm_init_()
{
    static_assert(in0_rows == 8 || in0_rows == 16 || in0_rows == 32, "dest_srca_mm: in0_rows must be 8, 16 or 32");
    _llk_math_matmul_init_<math_fidelity>(in0_rows, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, in0_rows == 8);

    // MOVD2A and the zero-flag clears address DEST and SrcA directly.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    // MOVD2A takes SrcA's format from this register only while implied formats are off.
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
}

/**
 * @brief Re-enable implied SrcA format after dest_srca_mm.
 */
inline void _llk_math_dest_srca_mm_uninit_()
{
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
}

/**
 * @brief First half of the math step of an unpack-to-DEST of a 32-bit tile: once earlier math is done, post the DEST
 * tile for the unpacker to write.
 *
 * @param dst_index: DEST tile the unpacker writes.
 * @note On the unpack thread, @ref _llk_unpack_A_ with unpack_to_dest copies the tile.
 * @note The copy is split into this post and @ref _llk_math_dest_srca_mm_unpack_wait_ so that math work can run while
 *       the unpacker copies the next tile. That work must not be an SFPLOADMACRO body with DEST stores that are not
 *       arbitrated against the unpacker's DEST writes (see sfpu_gguf_decode_body_unarbitrated).
 */
inline void _llk_math_dest_srca_mm_unpack_post_(const std::uint32_t dst_index)
{
    math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
}

/**
 * @brief As @ref _llk_math_dest_srca_mm_unpack_post_, with the 16-bit DEST rows of a page's block-float codes first
 * (@ref _llk_unpack_dest_srca_mm_codes_): Dst16b rows from codes_row, which must be a multiple of 4 from 4.
 */
inline void _llk_math_dest_srca_mm_codes_post_(const std::uint32_t dst_index, const std::uint32_t codes_row)
{
    math_unpack_to_dest_math_ready();
    mailbox_write(ThreadId::UnpackThreadId, (codes_row - 4) / 2);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
}

/**
 * @brief Second half: wait for the copy, then clear the tile's DEST zero flags again (Blackhole, budabackend#2730).
 *
 * @param dst_index: DEST tile posted by @ref _llk_math_dest_srca_mm_unpack_post_.
 * @param num_faces: Faces of the unpacked tile, values = <1/2/4>
 * @note A copy must not overlap @ref _llk_math_dest_srca_mm_: while it runs, the unpacker has SrcA's address swizzle
 *       off. Wait for it first.
 */
inline void _llk_math_dest_srca_mm_unpack_wait_(const std::uint32_t dst_index, const std::uint32_t num_faces)
{
    math::math_unpack_to_dest_tile_ready();
    // The math offset selects the bank, the index the tile within it.
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    for (std::uint32_t i = 0; i < num_faces; i++)
    {
        TT_ZEROACC(p_zeroacc::CLR_16, 1, 1 /*clear zero flags*/, ADDR_MOD_3, get_dest_index_in_faces(dst_index & 3, i));
    }
}

/**
 * @brief D[dst_index] += in0 * DEST tile src_index.
 *
 * Moves DEST tile src_index into SrcA, truncated to TF32 (SrcA's format with implied formats off), then runs the matmul
 * MOP into DEST tile dst_index and releases SrcB. in0 (SrcB) is [in0_rows, 32] (8 or 16 rows: its top two faces), in1
 * one [32, 32] Float32 DEST tile, the output one [in0_rows, 32] block of a Float32 DEST tile. Needs a 32-bit DEST; any
 * fidelity.
 *
 * @tparam math_fidelity: Must match the init.
 * @tparam in0_rows: Must match the init.
 * @param src_index: DEST tile of in1, Float32.
 * @param dst_index: DEST tile of the output.
 * @param src_rows: DEST rows past tile src_index where in1 starts (a tile at a row offset, as the GGUF decoder writes
 *                  for several pages in one tile).
 * @note The MVMUL sequence is recorded into the replay buffer on every call: SFPU code run since the last call may
 *       have recorded over it (the sfpi compiler turns unrolled loops into replays).
 * @note On the unpack thread, pair with @ref _llk_unpack_dest_srca_mm_, which hands the SrcA bank to this thread.
 */
template <MathFidelity math_fidelity, std::uint32_t in0_rows>
inline void _llk_math_dest_srca_mm_(const std::uint32_t src_index, const std::uint32_t dst_index, const std::uint32_t src_rows = 0)
{
    // The SrcA bank from the unpacker, and in1 written by the SFPU or math before it.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::MATH | p_stall::SRCA_VLD);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (src_index << 6) + src_rows + get_dest_buffer_base());
    math::reset_counters(p_setrwc::SET_ABD_F);
#pragma GCC unroll 16
    for (std::uint32_t r = 0; r < 4 * FACE_R_DIM; r += 4)
    {
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + r, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, r);
    }

    // The MVMULs of matmul_configure_mop for this in0, one in1 tile, SrcA reused.
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    constexpr auto end_clear     = high_fidelity ? p_setrwc::CLR_NONE : p_setrwc::CLR_A;
    constexpr std::uint32_t len  = in0_rows == 8 ? 4 : in0_rows == 16 ? 8 : 16;
    load_replay_buf(
        ckernel::math::replay_buf_offset,
        len,
        []
        {
            if constexpr (in0_rows == 8)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A2
            }
            else if constexpr (in0_rows == 16)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3
            }
            TTI_MVMUL(end_clear, 0, ADDR_MOD_5, 0); // the last one; next fidelity phase
        });

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
