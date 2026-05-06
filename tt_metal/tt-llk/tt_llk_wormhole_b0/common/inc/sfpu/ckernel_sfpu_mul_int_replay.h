// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "lltt.h"

namespace ckernel
{
// namespace sfpu
// {

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU integer multiplication: INT32 / UINT32 / UINT16.
//
// Mirrors the FP multiply replay path (`mul_binary_tile_replay` above), but
// for the integer dispatch normally served by `mul_int_tile<DataFormat>`.
// The recorded body is the per-iteration sequence of the upstream:
//   * INT32 / UINT32: `ckernel::sfpu::mul_int32`
//                     (sfpu/ckernel_sfpu_mul_int32.h, 30 instructions/iter)
//   * UINT16        : DISABLE_SFPLOADMACRO branch of
//                     `ckernel::sfpu::_mul_int_`
//                     (sfpu/ckernel_sfpu_mul_int.h, 20 instructions/iter)
//
// Both bodies depend on programmable LREG12-14 constants set by the upstream
// `mul_int_tile_init<DataFormat>` (which in turn calls `mul_int32_init` /
// `_init_mul_int_`):
//   INT32 / UINT32: vConstIntPrgm0 = 0x7ff, vConstIntPrgm1 = -11,
//                   vConstFloatPrgm2 = 8388608.0
//   UINT16        : vConstIntPrgm0 = 0xff, vConstIntPrgm1 = -8
//
// For UINT16, the non-DISABLE_SFPLOADMACRO build of `_init_mul_int_` instead
// programs `vConstFloatPrgm1 = 8388608.0` (LREG13) for the macro pipeline,
// which would break the discrete SFPSHFT2 used by this body. The
// `_llk_math_eltwise_binary_sfpu_init_uint16_replay_` helper therefore re-asserts
// `vConstIntPrgm1 = -8` before recording, so the same discrete body is
// correct in both build modes.
//
// Both bodies advance dst_reg by 2 rows via SFPSTORE with `ADDR_MOD_2`,
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

// INT32 / UINT32 (30 instructions/iter):
//   SFPLOAD (in0)
//   + SFPSHFT2 x2 (a1 raw, a2 raw)
//   + SFPAND + SFPCAST + SFPCAST + SFPAND + SFPCAST  (a0/a1/a2 -> fp32 chunks)
//   + SFPLOAD (in1)
//   + SFPSHFT2 x2 (b1 raw, b2 raw)
//   + SFPCAST                                       (b2 -> fp32)
//   + SFPMAD                                        (top  = a0*b2 + 2**23)
//   + SFPAND + SFPCAST                              (b1 -> fp32)
//   + SFPMAD                                        (top += a1*b1)
//   + SFPAND + SFPCAST                              (b0 -> fp32)
//   + SFPMAD                                        (top += a2*b0)
//   + SFPMAD                                        (mid = a0*b1 + 2**23)
//   + SFPMAD                                        (low = a0*b0 + 2**23)
//   + SFPMAD                                        (mid += a1*b0)
//   + SFPEXMAN x3                                   (extract integer mantissas)
//   + SFPSHFT x2                                    (top <<= 22, mid <<= 11)
//   + SFPIADD x2                                    (low += mid + top)
//   + SFPSTORE                                      (auto-advance via ADDR_MOD_2)
constexpr std::uint32_t SFPU_BINARY_MUL_INT32_REPLAY_LEN = 30;

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
//   + SFPSTORE                                      (auto-advance via ADDR_MOD_2)
constexpr std::uint32_t SFPU_BINARY_MUL_UINT16_REPLAY_LEN = 20;

// One-iteration body of `ckernel::sfpu::mul_int32`. Inputs are split into
// 11-bit chunks (a0/a1/a2, b0/b1/b2), promoted to fp32 (lossless), multiplied
// via three FMA streams (top/mid/low), then reassembled into a 32-bit int via
// SFPEXMAN + SFPSHFT + SFPIADD. Bit-exact with the upstream non-replay path.
ALWI void _llk_math_eltwise_binary_sfpu_init_int32_replay_()
{
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_INT32_REPLAY_LEN);

    // a -> a0 (LREG0), a1 (LREG2), a2 (LREG4) as fp32. LREG12 = 0x7ff,
    // LREG13 = -11 (logical right shift amount).
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // a >> 11
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // a >> 22
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);                                           // a1 = (a >> 11) & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);                                              // a1 -> fp32
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);                                              // a2 -> fp32 (top 10 bits, no mask)
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                           // a0 = a & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);                                              // a0 -> fp32

    // b -> b0 (LREG1), b1 (LREG3), b2 (LREG5) as fp32.
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0); // b2 -> fp32

    // top = a0*b2 + 2**23 (LREG14 = 8388608.0). FMA writes back into LREG5.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0); // b1 = (b >> 11) & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);    // b1 -> fp32
    // top += a1*b1
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0); // b0 = b & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);    // b0 -> fp32
    // top += a2*b0
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // mid = a0*b1 + 2**23 (into LREG6)
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);
    // low = a0*b0 + 2**23 (overwrites LREG0)
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);
    // mid += a1*b0
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    // Extract the integer mantissas (PAD9 zero-extends bit 23 = 0).
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);

    // SFPSHFT mod=1 = immediate-shift, logical fill (matches the rest of this
    // file's `TTI_SFPSHFT(..., 1)` usage; the named sfpi constants for SHIFT
    // mode use a different encoding convention than the SFPSHFT op).
    TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);                                     // top <<= 22
    TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);                                     // mid <<= 11
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // low += mid
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // low += top

    // ADDR_MOD_2 (= entry 6 in the FPU table, configured by mul_int_tile_init
    // with .dest.incr = 2) auto-advances dst_reg by 2 rows after the store.
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
}

// One-iteration body of `ckernel::sfpu::_mul_int_` (DISABLE_SFPLOADMACRO
// branch). 16-bit inputs split into two 8-bit chunks each; partial products
// are computed in fp32 (each fits in 16 bits before rounding) and stitched
// back via SFP_STOCH_RND (FP32 -> u16) + SFPIADD + SFPSHFT.
ALWI void _llk_math_eltwise_binary_sfpu_init_uint16_replay_()
{
    // Force the discrete-body LREG13 constant. The upstream non-DISABLE init
    // sets vConstFloatPrgm1 = 8388608.0 (LREG13) for the macro pipeline,
    // which would break the SFPSHFT2 below; this re-asserts the value used
    // by the DISABLE_SFPLOADMACRO branch. Safe in both build modes.
    sfpi::vConstIntPrgm1 = -8;

    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_UINT16_REPLAY_LEN);

    // a -> a0 (LREG0), a1 (LREG2) as fp32. LREG12 = 0xff, LREG13 = -8.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SHFT_LREG); // a >> 8
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);                                              // a1 -> fp32
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                           // a0 = a & 0xff
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);                                              // a0 -> fp32

    // b -> b0 (LREG1), b1 (LREG3) as fp32.
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
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

    // ADDR_MOD_2 auto-advance (see int32 body comment above). LO16 stores the
    // low 16 bits of LREG0 (which now holds the combined u16 product).
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_2, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
}

// Shared per-tile loop. The existing mul / bitwise / shift paths each have
// their own near-identical copy of this loop; this helper collapses that
// duplication for the comparison ops by templating on the replay length.
// Behaviour is identical to those copies: idst0 seeds the dst_reg base, the
// recorded body is replayed 8 times per face for each of the 4 faces, and
// the face boundary advances dst_reg by 16 rows via two SETRWC pulses.
template <std::uint32_t REPLAY_LEN>
ALWI void _llk_math_eltwise_binary_sfpu_run_replay_(std::uint32_t idst0)
{
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++)
    {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++)
        {
            lltt::replay(0, REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

// } // namespace sfpu
} // namespace ckernel
