// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU integer multiplication: INT32 / UINT32 / UINT16.
//
// Mirrors the FP multiply replay path (`mul_binary_tile_replay` above), but
// for the integer dispatch normally served by `mul_int_tile<DataFormat>`.
// The recorded body is the per-iteration sequence of the upstream:
//   * INT32 / UINT32 (Blackhole): discrete SFPMUL24 sequence matching the
//     `DISABLE_SFPLOADMACRO` branch of `sfpu::mul_int32` (`ckernel_sfpu_mul_int32.h`).
//     (Default builds use the macro pipeline for non-replay `mul_int32`; macro ops are not
//      reliably replayable — recording them can hang the math thread.)
//   * UINT16        : DISABLE_SFPLOADMACRO branch of
//                     `ckernel::sfpu::_mul_int_`
//                     (sfpu/ckernel_sfpu_mul_int.h, 20 instructions/iter)
//
// INT32 / UINT32 depend on `mul_int_tile_init` -> `mul_int32_init` (macro
// templates + templates on Blackhole when SFPLOADMACRO is enabled).
//
// UINT16 init notes:
//   UINT16        : vConstIntPrgm0 = 0xff, vConstIntPrgm1 = -8
//
// For UINT16, the non-DISABLE_SFPLOADMACRO build of `_init_mul_int_` instead
// programs `vConstFloatPrgm1 = 8388608.0` (LREG13) for the macro pipeline,
// which would break the discrete SFPSHFT2 used by this body. The
// `_init_replay_binary_sfpu_mul_uint16_` helper therefore re-asserts
// `vConstIntPrgm1 = -8` before recording, so the same discrete body is
// correct in both build modes.
//
// Both bodies advance dst_reg by 2 rows via SFPSTORE with `ADDR_MOD_6`,
// which `mul_int_tile_init` configures with `.dest.incr = 2`. No explicit
// INCRWC is needed inside the recorded body. Eight replays per face leave
// dst_reg advanced by 16 rows (= one face); the surrounding loop then
// advances another 16 rows via two SETRWC pulses, matching the upstream
// non-replay layout.
//
// Layout (offsets in dest rows, relative to current dst_reg base):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1])
//   result overwrites offset 0
// matching the kernel pairing `mul_int_tile(i*2, i*2 + 1, i*2)`.
// ---------------------------------------------------------------------------

constexpr std::uint32_t SFPU_BINARY_MUL_INT_DST_TILE_ROWS = 64;

// Discrete SFPMUL24 body (same as `mul_int32` when DISABLE_SFPLOADMACRO is set in
// `ckernel_sfpu_mul_int32.h`). Replay hardware records a linear SFPU insn stream;
// SFPLOADMACRO-based sequences from the default macro path are not safe to record/replay
// and can hang the math thread — use this discrete body for replay init only.
constexpr std::uint32_t SFPU_BINARY_MUL_INT32_REPLAY_LEN = 13;

// UINT16 discrete body (20 instructions/iter):
//   SFPLOAD (in0)
//   + SFPSHFT2 + SFPCAST + SFPAND + SFPCAST         (a1 / a0 -> fp32)
//   + SFPLOAD (in1)
//   + SFPSHFT2 + SFPCAST + SFPAND + SFPCAST         (b1 / b0 -> fp32)
//   + SFPMAD x3                                     (hi0=a0*b1, lo=a0*b0, hi1=a1*b0)
//   + SFP_STOCH_RND x3                              (round to u16)
//   + SFPIADD                                       (hi = hi0 + hi1)
//   + SFPSHFT                                       (hi <<= 8)
//   + SFPIADD                                       (lo += hi)
//   + SFPSTORE                                      (auto-advance via ADDR_MOD_6)
constexpr std::uint32_t SFPU_BINARY_MUL_UINT16_REPLAY_LEN = 20;

// Blackhole int32 multiply replay: must stay aligned with
// `hw/ckernels/blackhole/.../ckernel_sfpu_mul_int32.h` (`mul_int32`).
ALWI void _init_replay_binary_sfpu_mul_int32_()
{
    constexpr std::uint32_t offset_in0 = 0;
    constexpr std::uint32_t offset_in1 = SFPU_BINARY_MUL_INT_DST_TILE_ROWS;
    constexpr std::uint32_t offset_out = 0;

    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_INT32_REPLAY_LEN);

    // Mirrors `#ifdef DISABLE_SFPLOADMACRO` branch of Blackhole `sfpu::mul_int32`
    // (see `ckernel_sfpu_mul_int32.h`): discrete SFPMUL24 multiply only.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset_in0);
    TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG1, 5);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, offset_in1);
    TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, 5);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG5, 1);
    TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT(23, p_sfpu::LREG5, p_sfpu::LREG5, 5);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_6, offset_out);
}

// One-iteration body of `ckernel::sfpu::_mul_int_` (DISABLE_SFPLOADMACRO
// branch). 16-bit inputs split into two 8-bit chunks each; partial products
// are computed in fp32 (each fits in 16 bits before rounding) and stitched
// back via SFP_STOCH_RND (FP32 -> u16) + SFPIADD + SFPSHFT.
ALWI void _init_replay_binary_sfpu_mul_uint16_()
{
    // Force the discrete-body LREG13 constant. The upstream non-DISABLE init
    // sets vConstFloatPrgm1 = 8388608.0 (LREG13) for the macro pipeline,
    // which would break the SFPSHFT2 below; this re-asserts the value used
    // by the DISABLE_SFPLOADMACRO branch. Safe in both build modes.
    sfpi::vConstIntPrgm1 = -8;

    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_UINT16_REPLAY_LEN);

    // a -> a0 (LREG0), a1 (LREG2) as fp32. LREG12 = 0xff, LREG13 = -8.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // a >> 8
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);                                              // a1 -> fp32
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                           // a0 = a & 0xff
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);                                              // a0 -> fp32

    // b -> b0 (LREG1), b1 (LREG3) as fp32.
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_7, 1 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // b >> 8
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);                                              // b1 -> fp32
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);                                           // b0 = b & 0xff
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);                                              // b0 -> fp32

    // hi0 = a0*b1; lo = a0*b0; hi1 = a1*b0 (all written back to operand reg).
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

    // FP32 -> UINT16 (each partial < 2**16).
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);

    // hi = hi0 + hi1; hi <<= 8; lo += hi.
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_CC_NONE);
    // SFPSHFT mod=1 = immediate-shift, logical fill (see int32 body comment).
    TTI_SFPSHFT(8, 0, p_sfpu::LREG2, 1);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

    // ADDR_MOD_6 auto-advance (see int32 body comment above). LO16 stores the
    // low 16 bits of LREG0 (which now holds the combined u16 product).
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_6, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
}

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU elementwise multiplication.
//
// The standard `mul_binary_tile` expands (per tile) to 4 faces x 8 sfpi
// iterations of compiled SFPU instructions. That sequence is identical from
// one tile to the next for a given precision mode; the only thing that needs
// to change per tile is the DST base. We therefore record one sfpi-iteration
// body once at init time into the MATH-thread replay buffer and replay it at
// runtime, matching the output of the FPU-based `BINARY_OP` (eltwise multiply)
// used in `eltwise_binary_dram_optimized.cpp`:
//   * multiply in FP32 inside the SFPU
//   * when dest is BF16, software RNE FP32 -> BF16 and clamp 0*x = x*0 = 0
//     so the result matches the FPU's BF16 multiply bit-exactly
//   * when dest is FP32, skip rounding/clamping
//
// Layout baked into the recorded body (offsets are in dest rows, relative to
// the current dst_reg base set by `_llk_math_eltwise_binary_sfpu_start_`):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1], next tile slot)
//   result overwrites offset 0
// The body ends with INCRWC(SFP_DESTREG_STRIDE) mirroring `sfpi::dst_reg++`.
//
// The per-tile wrapper preserves the 4-face / 2 x SETRWC face-advance
// structure of `_llk_math_eltwise_binary_sfpu_params_` but seeds the DST
// base with `idst0` so the fixed offsets land on DST[idst0] and
// DST[idst0 + 1]. Callers must therefore use the same pairing the kernel
// already relies on: `replay_binary_sfpu_mul_float(i*2, i*2 + 1, i*2)`.
// ---------------------------------------------------------------------------

// Number of instructions in one recorded sfpi-iteration body.
//
// Two implementations are supported, selected at compile time:
//
//   DISABLE_SFPLOADMACRO defined  -> discrete SFPU instructions (SFPLOAD x2,
//                                    SFPMUL, etc.). Used as a portability /
//                                    debugging fallback or on builds where
//                                    the SFPLOADMACRO unit is unavailable.
//     is_fp32_dest_acc_en = true  : SFPLOAD x2 + SFPMUL + SFPNOP + SFPSTORE
//                                   + INCRWC = 6
//     is_fp32_dest_acc_en = false : software RNE FP32 -> BF16
//                                   + zero clamp = 20
//
//   DISABLE_SFPLOADMACRO undefined -> FP32 keeps the SFPLOADMACRO fused
//                                     LD + MAD + STORE path. BF16 uses the
//                                     discrete software-RNE body because it
//                                     matches the FPU path bit-exactly.
//     is_fp32_dest_acc_en = true  : SFPLOAD + SFPLOADMACRO + INCRWC = 3
//                                   (macro pipeline does LD + MAD + STORE)
//     is_fp32_dest_acc_en = false : use the same discrete software-RNE body
//                                   as the DISABLE_SFPLOADMACRO path = 20
#ifdef DISABLE_SFPLOADMACRO
constexpr std::uint32_t SFPU_BINARY_MUL_REPLAY_LEN = DST_ACCUM_MODE ? 6 : 20;
#else
constexpr std::uint32_t SFPU_BINARY_MUL_REPLAY_LEN = DST_ACCUM_MODE ? 3 : 20;
#endif

// A 32x32 tile occupies 64 rows in dest.
constexpr std::uint32_t SFPU_BINARY_MUL_DST_TILE_ROWS = 64;

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_replay_binary_sfpu_mul_float_()
{
#ifndef DISABLE_SFPLOADMACRO
    // ----- One-time SFPLOADMACRO setup (NOT recorded into the replay buffer) -----
    //
    // Each SFPLOADMACRO call performs:
    //   1. LD  : load from dst[base + dest_reg_addr] into a loaded LREG
    //            (loaded LREG = lreg_ind[1:0], so LREG[0..3] only)
    //   2. Pipeline : run the slots scheduled by Macro Sequence Register
    //                 lreg_ind[3:2] (SIMPLE / MAD / ROUND / STORE).
    //
    // Slot bit layout (per typecast/exp init patterns):
    //   bit 7 : UsesLoadValAsSrcB (bit 6 src for SIMPLE) -- override the
    //           instruction's srcB with the loaded LREG
    //   bit 6 : UsesStaging        -- redirect the instruction's dest to
    //           LREG[16] (staging) instead of the loaded LREG. We do not need
    //           staging here: MAD writes the product back into the loaded
    //           LREG, which the next stage then consumes.
    //   bits 5:3 : delay (cycles after LD before this slot fires)
    //   bits 2:0 : macro instruction mux index
    //              (3 = fixed STORE, 4..7 = programmable templates 0..3)
    //
    // Programmable Macro Instruction Template 1 (mux 5): SFPMAD
    //   lreg_dest = 13 selects backdoor-load into template 1.
    //   When triggered by SFPLOADMACRO with mad_bits bit 7 = 1 and bit 6 = 0:
    //     loaded_LREG = LREG[0] * loaded_LREG + LCONST_0
    //                 = RHS * LHS = LHS * RHS
    TTI_SFPMAD(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 13, 0);

    // Macro Sequence Register 0:
    //   simple slot: disabled
    //   MAD slot   : template 1 (SFPMAD) at delay 0; bit 7 = 1 selects loaded
    //                LREG as srcB; bit 6 = 0 writes the result back to the
    //                loaded LREG (no staging).
    //   round slot : disabled. BF16 does not use this macro sequence.
    //   store slot : FP32 mode -> fixed STORE (mux 3) at delay 2. Reads the
    //                loaded LREG (= MAD result) and writes it to the LD's
    //                dest_reg_addr. This replaces the discrete SFPSTORE.
    //                BF16 mode -> disabled.
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = is_fp32_dest_acc_en ? (0x00u | 0x00u | (2u << 3) | 3u) : 0u;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc config: UnitDelayKind[0] = 1 (WaitForElapsedInstructions, prevents
    // pipeline advancement on dest bank conflicts). StoreMod0 = DEFAULT lets
    // the store inherit the dst's natural format (matches the original
    // SFPSTORE's InstrModLoadStore::DEFAULT).
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
#endif // !DISABLE_SFPLOADMACRO

    // ----- Recorded per-iteration replay body -----
    //
    // Layout (offsets in dest rows, relative to the current dst_reg base):
    //   operand A (LHS)  at offset 0
    //   operand B (RHS)  at offset 64 (= 1 * SFPU_BINARY_MUL_DST_TILE_ROWS)
    //   result overwrites offset 0
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_REPLAY_LEN);

#ifdef DISABLE_SFPLOADMACRO
    // -----------------------------------------------------------------------
    // Discrete-instruction fallback (DISABLE_SFPLOADMACRO defined).
    // -----------------------------------------------------------------------
    // Loads both operands explicitly, multiplies them with SFPMUL, and stores
    // the result. This is the original implementation, preserved as a
    // portability path for builds where SFPLOADMACRO is unavailable or
    // disabled for debugging.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPNOP;

    if constexpr (!is_fp32_dest_acc_en)
    {
        // Bitwise RNE FP32 -> BF16: (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000.
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 2);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPSHFT_MOD1_SHIFT_LREGC);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 1);
        TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG3, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 0x7FFF);
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0xFFFF);
        TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG4, 0);

        // Match the FPU multiply convention: zero if either input is zero.
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    }

    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }
    else
    {
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }
#else
    // -----------------------------------------------------------------------
    // SFPLOADMACRO-fused implementation.
    // -----------------------------------------------------------------------
    if constexpr (!is_fp32_dest_acc_en)
    {
        // BF16 uses the same discrete body as the DISABLE_SFPLOADMACRO path so
        // software RNE consumes the SFPMUL result with the same instruction timing.
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        TTI_SFPNOP;

        // Bitwise RNE FP32 -> BF16: (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000.
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG1, 2);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT_MOD1_SHIFT_LREGC);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 1);
        TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x7FFF);
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xFFFF);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

        // Match the FPU multiply convention: zero if either input is zero.
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }
    else
    {
        // Pre-load operand B (RHS) at offset 64 into LREG[0]. The MAD template
        // installed above uses LREG[0] as srcA, so this value participates in the
        // multiply executed inside the SFPLOADMACRO pipeline.
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);

        // Issue SFPLOADMACRO: macro_idx = 0, loaded LREG = LREG[1], offset = 0.
        //   FP32 : LD LHS -> LREG[1]; MAD LREG[0]*LREG[1] -> LREG[1]; STORE LREG[1] to offset 0.
        TTI_SFPLOADMACRO((0 << 2) | 1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }
#endif // DISABLE_SFPLOADMACRO

    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

} // namespace ckernel
