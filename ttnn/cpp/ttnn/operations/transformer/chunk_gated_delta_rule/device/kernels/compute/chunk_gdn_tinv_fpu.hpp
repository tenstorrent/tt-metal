// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WY inverse of one 32x32 tile on the FPU, chained through DST (Blackhole, fp32 DST).
//   T_inv = (I - negN)^-1 for a strictly-lower negN = [[N00, 0], [N10, N11]] (16-quadrants):
//   Bi00 = (I - N00)^-1, Bi11 = (I - N11)^-1 by a 16-term power series per block, off = Bi11 @ N10 @ Bi00,
//   T_inv = [[Bi00, 0], [off, Bi11]]. Operands move DST -> SrcA/SrcB with MOVD2A/MOVD2B (tf32 truncation, the
//   Src format pinned to tf32 meanwhile), products accumulate in fp32 DST at HiFi4; nothing touches L1 between
//   the first unpack and the final pack. Defaults: HornerR (S in SrcB), negN datacopied to DST.

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

namespace gdn_tinv_fpu {

// How negN reaches the source registers.
enum class NSrc : uint8_t {
    Unpack,  // matmul-unpacked into SrcA and parked in DST tile 3 by MOVA2D; the banks stay with MATH
    Dst,     // datacopied into DST tile 3; the banks come from an unpacker dummy-valid
};
// Series form inside each 16-block.
enum class Form : uint8_t {
    Square,   // S_2m = S_m + S_m P_m, P_2m = P_m^2: 4 doublings
    Horner,   // S_m = I + N S_{m-1}: 15 steps, N in SrcB, S in SrcA (the op order of invert16)
    HornerR,  // S_m = I + S_{m-1} N: 15 steps, S in SrcB, N in SrcA
};

// DST tile map (fp32 half-sync bank: 4 tiles of 64 rows).
constexpr uint32_t kTout = 0;  // I at entry, T_inv at exit
constexpr uint32_t kTp1 = 1;   // power scratch
constexpr uint32_t kTp2 = 2;   // power scratch, tmp (face 2)
constexpr uint32_t kTn = 3;    // negN (faces 0, 2, 3)

constexpr uint32_t row(uint32_t tile, uint32_t face, uint32_t r = 0) { return tile * 64 + face * 16 + r; }
constexpr uint32_t kFmtDefault = 3;  // pin SrcA and SrcB to tf32 for the chain (detail::kFmtSrcA | kFmtSrcB)

#ifdef TRISC_MATH
namespace detail {
using namespace ckernel;

// DST rows D..D+15 -> SrcB rows B..B+15 (4 rows per instruction, tf32 truncation).
template <uint32_t B, uint32_t D>
inline void d2b() {
    TTI_MOVD2B(0, B + 0, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, D + 0);
    TTI_MOVD2B(0, B + 4, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, D + 4);
    TTI_MOVD2B(0, B + 8, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, D + 8);
    TTI_MOVD2B(0, B + 12, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, D + 12);
}
// DST rows D..D+15 -> SrcA rows A..A+15.
template <uint32_t A, uint32_t D>
inline void d2a() {
    TTI_MOVD2A(0, A + 0, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, D + 0);
    TTI_MOVD2A(0, A + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, D + 4);
    TTI_MOVD2A(0, A + 8, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, D + 8);
    TTI_MOVD2A(0, A + 12, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, D + 12);
}
// SrcA rows A..A+15 -> DST rows D..D+15.
template <uint32_t A, uint32_t D>
inline void a2d() {
    TTI_MOVA2D(0, A + 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, D + 0);
    TTI_MOVA2D(0, A + 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, D + 8);
}
// DST rows D..D+7 += SrcB[rwc_b..+7] @ SrcA[rwc_a face], then apply MOD.
template <uint32_t MOD, uint32_t D>
inline void mv() {
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, MOD, D);
}
// Mark one 16-row face of the current bank undefined (reads as zero).
template <uint32_t FACE>
inline void zero_face() {
    TTI_ZEROACC(p_zeroacc::CLR_16, 1, 0, ADDR_MOD_7, FACE);
}
template <bool STALL>
inline void stage_barrier() {
    if constexpr (STALL) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);
    }
}

// ADDR_MOD_4/5: counters back to 0, fidelity phase +1 / cleared. ADDR_MOD_7: no-op (the moves).
inline void set_common_mods() {
    addr_mod_t{.srca = {.clr = 1, .cr = 1}, .srcb = {.clr = 1, .cr = 1}, .fidelity = {.incr = 1}}.set(ADDR_MOD_4);
    addr_mod_t{.srca = {.clr = 1, .cr = 1}, .srcb = {.clr = 1, .cr = 1}, .fidelity = {.clr = 1}}.set(ADDR_MOD_5);
    addr_mod_t{.srcb = {.incr = 8}}.set(ADDR_MOD_6);
    addr_mod_t{}.set(ADDR_MOD_7);
}

// DST rows D..D+15 += SrcB rows 0..15 @ SrcA rows 0..15 at HiFi4. Counters 0 on entry and exit.
template <uint32_t D>
inline void face_mm() {
    mv<ADDR_MOD_6, D>();
    mv<ADDR_MOD_4, D + 8>();
    mv<ADDR_MOD_6, D>();
    mv<ADDR_MOD_4, D + 8>();
    mv<ADDR_MOD_6, D>();
    mv<ADDR_MOD_4, D + 8>();
    mv<ADDR_MOD_6, D>();
    mv<ADDR_MOD_5, D + 8>();
}

// off = Bi11 @ N10 @ Bi00 into kTout face 2 (zero on entry); Bi00/Bi11 in kTout faces 0/3, N10 in DST rows N10.
template <uint32_t N10>
inline void off_diagonal() {
    d2b<0, row(kTout, 3)>();
    d2a<0, N10>();
    face_mm<row(kTp2, 2)>();
    d2b<0, row(kTp2, 2)>();
    d2a<0, row(kTout, 0)>();
    face_mm<row(kTout, 2)>();
}

// ---- Square: both blocks per stage. SrcB: P00 0..15, S00 16..31, S11 32..47, P11 48..63; SrcA: P00 0..15, P11 48..63.
inline void set_square_mods() {
    addr_mod_t{.srcb = {.incr = 16}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 48}, .srcb = {.incr = 32}}.set(ADDR_MOD_1);
    addr_mod_t{.srcb = {.incr = 48}}.set(ADDR_MOD_2);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 40}}.set(ADDR_MOD_3);
    set_common_mods();
}
// One fidelity phase: S_b += S_b P_b (kTout faces 0/3), P_b^2 into P0/P1 (DST rows).
template <uint32_t P0, uint32_t P1, uint32_t END>
inline void square_phase() {
    mv<ADDR_MOD_0, P0>();
    mv<ADDR_MOD_1, row(kTout, 0)>();
    mv<ADDR_MOD_2, P1>();
    mv<ADDR_MOD_3, row(kTout, 3)>();
    mv<ADDR_MOD_0, P0 + 8>();
    mv<ADDR_MOD_1, row(kTout, 0, 8)>();
    mv<ADDR_MOD_2, P1 + 8>();
    mv<END, row(kTout, 3, 8)>();
}
template <uint32_t P0, uint32_t P1>
inline void square_stage() {
    square_phase<P0, P1, ADDR_MOD_4>();
    square_phase<P0, P1, ADDR_MOD_4>();
    square_phase<P0, P1, ADDR_MOD_4>();
    square_phase<P0, P1, ADDR_MOD_5>();
}
// Load the next doubling's operands: powers from DST rows Q0/Q1, sums from kTout.
template <uint32_t Q0, uint32_t Q1>
inline void square_load() {
    d2b<0, Q0>();
    d2b<16, row(kTout, 0)>();
    d2b<32, row(kTout, 3)>();
    d2b<48, Q1>();
    d2a<0, Q0>();
    d2a<48, Q1>();
}

// NSrc::Unpack: the unpacker left negN in SrcA; park its three live faces in DST tile kTn. Every Src row the
// MVMULs read is then written by MOVD2A/MOVD2B (an unpacker-written SrcA face is misread once MOVD2B has run).
template <NSrc S>
inline void park_n() {
    if constexpr (S == NSrc::Unpack) {
        a2d<0, row(kTn, 0)>();
        a2d<32, row(kTn, 2)>();
        a2d<48, row(kTn, 3)>();
    }
}

template <NSrc S, bool STALL>
inline void square_chain() {
    set_square_mods();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    park_n<S>();
    d2b<0, row(kTn, 0)>();
    d2b<16, row(kTout, 0)>();
    d2b<32, row(kTout, 3)>();
    d2b<48, row(kTn, 3)>();
    d2a<0, row(kTn, 0)>();
    d2a<48, row(kTn, 3)>();
    square_stage<row(kTp1, 0), row(kTp1, 1)>();  // S2, N^2
    stage_barrier<STALL>();
    square_load<row(kTp1, 0), row(kTp1, 1)>();
    square_stage<row(kTp1, 2), row(kTp1, 3)>();  // S4, N^4
    stage_barrier<STALL>();
    square_load<row(kTp1, 2), row(kTp1, 3)>();
    square_stage<row(kTp2, 0), row(kTp2, 1)>();  // S8, N^8
    stage_barrier<STALL>();
    square_load<row(kTp2, 0), row(kTp2, 1)>();
    square_stage<row(kTp1, 0), row(kTp1, 1)>();  // S16; the N^16 products land on dead faces
    stage_barrier<STALL>();
    off_diagonal<row(kTn, 2)>();
}

// ---- Horner: S_m = N S_{m-1} + I per block. SrcB: N00 0..15, I 16..31, N11 48..63; SrcA: S00 0..15, I 16..31,
// S11 48..63. HornerR swaps the roles: SrcB holds S00 / I / S11, SrcA holds N00 / I / N11 (same rows, same counter
// walk).
inline void set_horner_mods() {
    addr_mod_t{.srca = {.incr = 48}, .srcb = {.incr = 48}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 24}}.set(ADDR_MOD_1);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 16}}.set(ADDR_MOD_2);
    set_common_mods();
}
template <uint32_t END>
inline void horner_phase() {
    mv<ADDR_MOD_0, row(kTout, 0)>();
    mv<ADDR_MOD_1, row(kTout, 3)>();
    mv<ADDR_MOD_0, row(kTout, 0, 8)>();
    mv<END, row(kTout, 3, 8)>();
}
template <bool S_IN_B>
inline void horner_step() {
    if constexpr (S_IN_B) {
        d2b<0, row(kTout, 0)>();
        d2b<48, row(kTout, 3)>();
    } else {
        d2a<0, row(kTout, 0)>();
        d2a<48, row(kTout, 3)>();
    }
    zero_face<0>();
    zero_face<3>();
    horner_phase<ADDR_MOD_4>();
    horner_phase<ADDR_MOD_4>();
    horner_phase<ADDR_MOD_4>();
    horner_phase<ADDR_MOD_5>();
    TTI_ZEROACC_ADDRMOD_ONLY(ADDR_MOD_2);  // counters to (16, 16): the identity rows
    mv<ADDR_MOD_7, row(kTout, 0)>();
    mv<ADDR_MOD_6, row(kTout, 3)>();
    mv<ADDR_MOD_7, row(kTout, 0, 8)>();
    mv<ADDR_MOD_5, row(kTout, 3, 8)>();
}

template <NSrc S, bool STALL, bool S_IN_B>
inline void horner_chain() {
    set_horner_mods();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    park_n<S>();
    if constexpr (S_IN_B) {
        d2a<0, row(kTn, 0)>();
        d2a<48, row(kTn, 3)>();
    } else {
        d2b<0, row(kTn, 0)>();
        d2b<48, row(kTn, 3)>();
    }
    d2b<16, row(kTout, 0)>();
    d2a<16, row(kTout, 0)>();
    for (uint32_t m = 1; m < 16; m++) {
        stage_barrier<STALL>();
        horner_step<S_IN_B>();
    }
    stage_barrier<STALL>();
    off_diagonal<row(kTn, 2)>();
}

// Src register format control around the chain (bit mask): 1 pin SrcA to tf32, 2 pin SrcB to tf32 (both via
// DISABLE_IMPLIED_SRC?_FMT), 4 keep the Src zero-substitution flag (no low-byte flush).
constexpr uint32_t kFmtSrcA = 1, kFmtSrcB = 2, kFmtKeepZero = 4;
template <uint32_t FMT>
inline void src_format_enter() {
    if constexpr (FMT != 0) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    }
    if constexpr (FMT & kFmtSrcA) {
        TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
    }
    if constexpr (FMT & kFmtSrcB) {
        TTI_SETC16(DISABLE_IMPLIED_SRCB_FMT_Base_ADDR32, 1);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(to_underlying(DataFormat::Tf32));
    }
    if constexpr (FMT & kFmtKeepZero) {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
}
// Back to the implied per-bank formats; the base SrcA/SrcB formats are left at Float32 (the fp32 operand setting).
template <uint32_t FMT>
inline void src_format_leave() {
    if constexpr (FMT != 0) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    }
    if constexpr (FMT & kFmtSrcA) {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));
        TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    }
    if constexpr (FMT & kFmtSrcB) {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(to_underlying(DataFormat::Float32));
        TTI_SETC16(DISABLE_IMPLIED_SRCB_FMT_Base_ADDR32, 0);
    }
    if constexpr (FMT & kFmtKeepZero) {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
        math::_invalidate_src_zero_flag_state_();
    }
}

// Full MATH-side inverse: DST offset at the bank base, counters 0, banks returned at the end.
template <Form F, NSrc S, bool STALL, uint32_t FMT = kFmtSrcA | kFmtSrcB>
inline void chain() {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    src_format_enter<FMT>();
    if constexpr (F == Form::Square) {
        square_chain<S, STALL>();
    } else {
        horner_chain<S, STALL, F == Form::HornerR>();
    }
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    src_format_leave<FMT>();
}
}  // namespace detail
#endif  // TRISC_MATH

// T_inv = (I - negN)^-1 of one tile.
//   negN   : fp32 CB whose front tile is -strictly_lower(N), front-waited by the caller.
//   cb_eye : CB with the identity tile at index 0.
//   out    : fp32 CB that receives T_inv (one tile reserved and pushed here).
// Uses DST tiles 0..3 of the acquired half; all three CBs must be fp32.
template <Form F = Form::HornerR, NSrc S = NSrc::Dst, bool STALL = false, uint32_t FMT = kFmtDefault>
inline void tinv(uint32_t negN, uint32_t cb_eye, uint32_t out) {
    cb_reserve_back(out, 1);
    reconfig_data_format_srca(cb_eye);  // the identity and negN copies read fp32 through SrcA
    pack_reconfig_data_format(out);
    copy_init(cb_eye);
    tile_regs_acquire();
    copy_tile(cb_eye, 0, kTout);
    if constexpr (S == NSrc::Unpack) {
        matmul_init(negN, negN);
        UNPACK((llk_unpack_AB_matmul(negN, negN, 0, 0)));
    } else {
        copy_tile(negN, 0, kTn);
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
    }
    MATH((detail::chain<F, S, STALL, FMT>()));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(kTout, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

}  // namespace gdn_tinv_fpu
