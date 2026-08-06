// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK math primitives for the H128 (1x128) Hadamard transform.
//
// Goal: Y = H_128 * x for a 1x128 input vector x. Reshape x as the top
// 8 rows of a 16x16 zero-padded X_pad; by the H_{2N}-of-half-zero-padded
// identity, (H_16 X_pad H_16)[0..7] row-major-flattens to H_128 * x.
//
// All operands are 1-face [16, 16] bfloat16 tiles; the math path issues
// MVMUL / MOVD2B directly with our own addrmods.
//
// Operand routing (matches Compute API & unpack):
//   h16   -> srcB face (0, 0) and srcA bank 1 — the entire face is H_16.
//   input -> srcA bank 0 face (0, 0) — top 8 rows = X_pad, rest zero pad.
//
// Per tile, on the math thread (only dst rows 0..7 of each face matter):
//   1. MM1   : dst.face1[0..7] = (H_16 X_pad)[0..7]; CLR_A flips srcA to
//              bank 1 (H_16, streamed by the unpack thread during MM1).
//   2. MOVD2B: copy the MM1 result (dst.face1 -> srcB rows 0..7).
//   3. MM2   : dst.face0[0..7] = (H_16 X_pad)[0..7] * H_16
//                              = H_128 x reshape (8, 16).
//   4. Normalize (normalize=true): SFPU scale dst.face0 by 1/sqrt(128).
//
// The MM1/MM2 dst-face split (face 1 then face 0) lets MM2 accumulate
// into a clean face, removing the ZEROACC that a same-face scheme needs.
//
// Fidelity: LoFi runs 1 MVMUL/pass; high fidelity runs 2 (4 total) to
// capture x at full bf16 precision. See _configure_addrmod_ for the
// fidelity-phase derivation.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace ckernel;

// Program the addrmods the custom narrow MOP issues by hand.
//
// ADDR_MOD_7 is a true no-op used by every non-fidelity instruction
// (MOVD2B, the SFPU load/store, the LoFi MVMULs): each carries explicit
// src/dst row args and runs once, so we never want an implicit advance.
//
// The other three slots exist only for high fidelity, where each matmul
// runs two MVMULs that accumulate partial mantissa products into the same
// dst. They leave src/dst untouched and only step the FPU fidelity-phase
// counter, which selects the mantissa halves (cumulative):
//   phase 0 = srcA_hi·srcB_hi
//   phase 1 = srcA_lo·srcB_hi   (extends srcA)
//   phase 2 = srcA_hi·srcB_lo   (extends srcB)
//   phase 3 = srcA_lo·srcB_lo   (unused)
// H_16 is a ±1 sign matrix (no low mantissa), so any phase touching its
// low half is identically zero and is skipped:
//   ADDR_MOD_0 (+1): MM1 (srcA=x, srcB=H_16) -> phases {0,1}, full x.
//   ADDR_MOD_1 (+2): MM2 (srcA=H_16, srcB=intermediate) -> {0,2}, full
//                    intermediate.
//   ADDR_MOD_2 (reset): zeroes the counter after each pass's 2nd MVMUL,
//                       before MM2 and the next tile.
template <MathFidelity math_fidelity = MathFidelity::HiFi4>
inline void _llk_math_hadamard_h128_configure_addrmod_()
{
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (is_high_fidelity(math_fidelity))
    {
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0},
        }
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 2, .clr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_2);
    }
}

// 1/sqrt(128) as float32 (= sqrt(2)/16 ≈ 0.08838834764831843).
// Precomputed as constexpr so kH128NormBits folds into the
// _sfpu_load_config32_ immediates at compile time.
static constexpr float kH128NormScale        = 0.08838834764831843f;
static constexpr std::uint32_t kH128NormBits = __builtin_bit_cast(std::uint32_t, kH128NormScale);

template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void _llk_math_hadamard_h128_init_()
{
    _llk_math_hadamard_h128_configure_addrmod_<math_fidelity>();
    if constexpr (normalize)
    {
        // Reset shared SFPU config registers (LREG12-14), then write
        // kH128NormScale into LREG12 (vConstFloatPrgm0). The value
        // persists across all 4 SFPU row groups for the program lifetime.
        sfpu::_init_sfpu_config_reg();
        sfpu::_sfpu_load_config32_(p_sfpu::LREG12, (kH128NormBits >> 16) & 0xFFFF, kH128NormBits & 0xFFFF);
    }
}

// One H128 transform on the tile at dst_index (see the file header for
// the algorithm and the unpack/dst-face scheme).
//
// Precondition: dst_index's tile is zeroed on acquire and used for one
// transform per acquire — MM2 relies on a clean face 0, and we never
// re-zero face 1 between calls.
template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void _llk_math_hadamard_h128_(std::uint32_t dst_index)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // dst row 16 = face 1 (MM1 target), row 0 = face 0 (MM2 target).

    // Position dst and zero the rwc (incl. the fidelity phase — SET_ABD_F
    // covers F) so the body addresses dst starting at phase 0.
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // MM1: dst.face1[0..7] = srcB * srcA = H_16 * X_pad = (H_16 X_pad)[0..7].
    // HiFi adds phase 1 (see _configure_addrmod_). The final MVMUL carries
    // CLR_A to release srcA bank 0 so the FPU advances to bank 1 (H_16 from
    // phase-2 unpack) for MM2. srcB is left valid (FPU keeps ownership).
    if constexpr (high_fidelity)
    {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 16);
        TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 16);
    }
    else
    {
        TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_7, 16);
    }

    // Drain MM1's dst writeback before MOVD2B reads dst: MVMUL writes dst
    // on the FPU pipe, MOVD2B reads it on the MOV pipe, which doesn't
    // auto-track the hazard. Without it MOVD2B races ahead and reads
    // pre-MM1 dst (PCC failures that vanish when DPRINTs throttle math).
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);

    // MOVD2B: dst.face1 rows 16..23 (= (H_16 X_pad)[0..7]) -> srcB rows 0..7,
    // overwriting H_16[0..7] in the FPU-owned srcB bank. srcB rows 8..15
    // keep H_16; MM2 reads only rows 0..7, so the residue is harmless.
    TTI_MOVD2B(0, 0, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 16);
    TTI_MOVD2B(0, 4, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 20);

    // Wait for MOVD2B to complete
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);

    // MM2: dst.face0[0..7] = srcB * srcA = (H_16 X_pad)[0..7] * H_16
    //                = H_128 x reshape (8, 16). srcA = H_16 (bank 1),
    // srcB = MM1 result (MOVD2B). HiFi adds phase 2 (see _configure_addrmod_).
    // face 0 is clean (acquire-zeroed, MM1 wrote face 1) so accumulation
    // starts from zero. The final MVMUL carries CLR_AB to release srcA/srcB
    // dvalid for the next tile's UNPACR — without it the dvalid flip-flops
    // persist across invocations and the next unpack deadlocks.
    if constexpr (high_fidelity)
    {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_2, 0);
    }
    else
    {
        TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_7, 0);
    }

    // Normalize: dst[0..7] *= kH128NormScale (LREG12 = 1/sqrt(128)).
    if constexpr (normalize)
    {
        // FPU->SFPU dst hazard: drain MM2's writeback before SFPU reads dst.
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

        // FP16B mode converts bfloat16 <-> float32 on load/store;
        // SFPMUL(A, B, LCONST_0, D) computes D = A*B. Issuing all 4 loads
        // before the muls (then all stores) hides the 4-cycle SFPU
        // pipeline latency — each group of 4 fills one pipeline depth.
        constexpr auto kFP16B = InstrModLoadStore::FP16B;

        TTI_SFPLOAD(p_sfpu::LREG0, kFP16B, ADDR_MOD_7, 0); // rows 0..1
        TTI_SFPLOAD(p_sfpu::LREG1, kFP16B, ADDR_MOD_7, 2); // rows 2..3
        TTI_SFPLOAD(p_sfpu::LREG2, kFP16B, ADDR_MOD_7, 4); // rows 4..5
        TTI_SFPLOAD(p_sfpu::LREG3, kFP16B, ADDR_MOD_7, 6); // rows 6..7

        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

        TTI_SFPSTORE(p_sfpu::LREG0, kFP16B, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, kFP16B, ADDR_MOD_7, 2);
        TTI_SFPSTORE(p_sfpu::LREG2, kFP16B, ADDR_MOD_7, 4);
        TTI_SFPSTORE(p_sfpu::LREG3, kFP16B, ADDR_MOD_7, 6);
    }
}

inline void _llk_math_hadamard_h128_uninit_()
{
    // Intentionally empty -- but NOT because the init left no state behind. _llk_math_hadamard_h128_init_
    // programs ADDR_MOD_0/1/2/7 and, under normalize, resets the shared SFPU config registers and writes
    // kH128NormScale into LREG12 (vConstFloatPrgm0). Neither is undone here:
    //   - addrmods: every math LLK reprograms the mods it needs in its own init (see the
    //     _configure_addrmod_ call at the top of each), so a stale Hadamard mod is overwritten before use.
    //   - LREG12: persists for the program lifetime by design; a later SFPU op that expects a different
    //     vConstFloatPrgm0 must set it itself. That is a real cross-op coupling, tracked separately.
    // Kept as a no-op so the API stays symmetric with the init and matches the blaze original; making it
    // restore state would emit instructions and change behavior.
}
