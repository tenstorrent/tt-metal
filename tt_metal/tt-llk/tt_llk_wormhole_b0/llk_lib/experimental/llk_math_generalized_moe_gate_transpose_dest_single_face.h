// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "lltt.h"

using namespace ckernel;

namespace ckernel
{

constexpr std::uint32_t transpose_dest_tile_offset = 64; // 1 tile x 64 rows per tile

// Configure address modifiers for single face transpose
template <bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = transpose_dest_tile_offset},
    }
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step0_configure_mop()
{
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 16);
    // Move 8 rows from DEST to SrcB interleaved (1 row at a time)
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
    TTI_MOVD2B(0, 18, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 1);
    TTI_MOVD2B(0, 20, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 2);
    TTI_MOVD2B(0, 22, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 3);
    TTI_MOVD2B(0, 24, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 4);
    TTI_MOVD2B(0, 26, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 5);
    TTI_MOVD2B(0, 28, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 6);
    TTI_MOVD2B(0, 30, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 7);

    // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
    TTI_MOVB2D(0, 16, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 0);
    TTI_MOVB2D(0, 18, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 1);
    TTI_MOVB2D(0, 20, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 2);
    TTI_MOVB2D(0, 22, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 3);
    TTI_MOVB2D(0, 24, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 4);
    TTI_MOVB2D(0, 26, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 5);
    TTI_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 6);
    TTI_MOVB2D(0, 30, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 7);

    std::uint32_t d2b_instr = lltt::replay_insn(math::replay_buf_offset, 8);
    std::uint32_t b2d_instr = lltt::replay_insn(math::replay_buf_offset + 8, 8);

    ckernel_template tmp(num_tiles, 1, d2b_instr, TT_OP_TRNSPSRCB);
    tmp.set_end_op(b2d_instr);
    tmp.program();
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step1_configure_mop()
{
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 11);
    // Move 4 rows from DEST to SrcB (4 rows at a time)
    // This will place the first 2 rows in the first 2 columns, and the last 2 rows in the last 2 columns
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, 28, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);

    TTI_TRNSPSRCB;

    // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
    TTI_MOVB2D(0, 16, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 0);
    TTI_MOVB2D(0, 18, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 1);
    TTI_MOVB2D(0, 20, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 2);
    TTI_MOVB2D(0, 22, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 3);
    TTI_MOVB2D(0, 24, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 4);
    TTI_MOVB2D(0, 26, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 5);
    TTI_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 6);
    TTI_MOVB2D(0, 30, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 7);
    std::uint32_t replay_instr = lltt::replay_insn(math::replay_buf_offset, 11);

    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

// step1_hi: clone of step1 with two knobs:
//   d2b_dst  = MOVD2B source DEST row (which 4 groups to read; post-step0 group g is at row g).
//   b2d_base = MOVB2D output DEST row base (where to write the resulting run; step1 uses 0).
// The output base lets the LOW half write its run to rows 8-15 (b2d_base=8) so it does NOT clobber
// the post-step0 groups 4-7 sitting at rows 4-7 — which the HIGH half (d2b_dst=4) then reads.
template <std::uint32_t num_tiles, bool is_32bit, std::uint32_t d2b_dst, std::uint32_t b2d_base>
inline void generalized_moe_gate_transpose_dest_single_face_step1_hi_configure_mop()
{
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 11);
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, d2b_dst);
    TTI_MOVD2B(0, 28, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, d2b_dst);

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 0);
    TTI_MOVB2D(0, 18, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 1);
    TTI_MOVB2D(0, 20, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 2);
    TTI_MOVB2D(0, 22, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 3);
    TTI_MOVB2D(0, 24, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 4);
    TTI_MOVB2D(0, 26, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 5);
    TTI_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_1_ROW, b2d_base + 6);
    TTI_MOVB2D(0, 30, ADDR_MOD_2, p_movb2d::MOV_1_ROW, b2d_base + 7);
    std::uint32_t replay_instr = lltt::replay_insn(math::replay_buf_offset, 11);

    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

// Plain (non-transposed) FPU copy of 4 DEST rows [src..src+3] -> [dst..dst+3], across the 3 data
// regions (scores/indices/bias, via num_tiles + the ADDR_MOD_2 base advance). Used to stash/restore
// data in rows 8-15 — which the SFPU merge cannot address (SFPU offsets >=8 wrap) but the FPU can —
// during the ungrouped two-half assembly. src/dst must be 4-row aligned (0,4,8,12).
template <std::uint32_t num_tiles, bool is_32bit, std::uint32_t src, std::uint32_t dst, std::uint32_t srcb = 16>
inline void generalized_moe_gate_copy4rows_configure_mop()
{
    static_assert(!is_32bit, "32-bit is not supported");
    // srcb selects the 4-row SrcB scratch window (16/20/24/28). Back-to-back copy4rows calls use
    // DISJOINT srcb windows so a later MOVB2D cannot read a previous copy's SrcB leftover.
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 5);
    TTI_MOVD2B(0, srcb, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, src);        // DEST rows src..+3 -> SrcB srcb..+3
    TTI_MOVB2D(0, srcb + 0, ADDR_MOD_3, p_movb2d::MOV_1_ROW, dst + 0); // SrcB srcb -> DEST dst+0 (no transpose)
    TTI_MOVB2D(0, srcb + 1, ADDR_MOD_3, p_movb2d::MOV_1_ROW, dst + 1);
    TTI_MOVB2D(0, srcb + 2, ADDR_MOD_3, p_movb2d::MOV_1_ROW, dst + 2);
    TTI_MOVB2D(0, srcb + 3, ADDR_MOD_2, p_movb2d::MOV_1_ROW, dst + 3); // ADDR_MOD_2 advances base by 64
    std::uint32_t replay_instr = lltt::replay_insn(math::replay_buf_offset, 5);
    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step2_configure_mop()
{
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 4);
    // Move 8 rows from DEST to SrcB (4 rows at a time)
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 4);

    TTI_TRNSPSRCB;

    // Move 1 row from SrcB back to DEST
    TTI_MOVB2D(0, 16, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 0);
    std::uint32_t replay_instr = lltt::replay_insn(math::replay_buf_offset, 4);

    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_()
{
    generalized_moe_gate_transpose_dest_single_face_configure_addrmod<is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step0_configure_mop<4, is_32bit>();
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step1_configure_mop<3, is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_()
{
    // num_tiles=3: transpose scores(0) + idx(1) + BIAS(2). The bias (tile 2) MUST be math->standard'd too,
    // else the combine's bias round-trip packs a math-layout bias -> 2-period-corrupted sort key (the 256
    // output path never reads bias, so num_tiles=2 was enough there; the >256 combine merge sorts by bias).
    generalized_moe_gate_transpose_dest_single_face_step2_configure_mop<3, is_32bit>();
}

// copy4rows init/runner.
template <std::uint32_t src = 0, std::uint32_t dst = 0, bool is_32bit = false, std::uint32_t srcb = 16>
inline void _llk_math_generalized_moe_gate_copy4rows_init_()
{
    generalized_moe_gate_copy4rows_configure_mop<3, is_32bit, src, dst, srcb>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_copy4rows_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit / fp32 dest accum not supported");
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);
    ckernel_template::run();
}

// step1_hi init/runner — tunable knobs (d2b_dst, b2d_base) for the high-group experiment.
template <std::uint32_t d2b_dst = 0, std::uint32_t b2d_base = 24, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step1_hi_configure_mop<3, is_32bit, d2b_dst, b2d_base>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);
    ckernel_template::run();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    // Run the 16-bit single-face transpose MOP
    ckernel_template::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    // Run the 16-bit single-face transpose MOP
    ckernel_template::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    ckernel_template::run();

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

} // namespace ckernel
