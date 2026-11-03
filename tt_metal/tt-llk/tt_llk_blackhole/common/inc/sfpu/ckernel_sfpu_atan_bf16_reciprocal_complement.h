// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_atan_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_atan_bf16_reciprocal_complement_core.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::atan
{
// Replay slots of one BH row: 12 ahead of the Horner rungs, two per rung (a gap slot,
// then the MAD), one more where degree seven reloads LREG3, and the 4-slot tail.
constexpr unsigned reciprocal_complement_body_slots(unsigned degree)
{
    return degree == 8 ? 32u : 12u + 2u * (degree - 1u) + (degree == 7 ? 1u : 0u) + 4u;
}

// SFPU state this kernel overwrites on Blackhole and does not restore. A later op that reads any of it
// must program it in its own init; one that skips its init, or relies on load-macro or replay state it
// did not program, runs with this kernel's.
// - init_reciprocal_complement: vConstFloatPrgm0 (2.0, sfpu_reciprocal_init) and the SFPLOADMACRO
//   configuration: the instruction templates written by the SFPSETSGN and SFP_STOCH_RND below, macro
//   sequence 0 (kMacroSequenceBits, SFPCONFIG 4) and the load-macro misc settings (SFPCONFIG 8).
// - calculate_reciprocal_complement, on every call: LREG4 (the complement constant), LREG5 (the NaN-row
//   bitmap), LREG6 and LREG7 (c1 and c2), LREG0-3 as scratch, and replay slots 0..kBodySlots+3 (the row
//   loads its two leading coefficients too), which it records again on each call. The DEST row counter is set back to row 0 of the tile.
// Each row loads and stores through ADDR_MOD_7, which every SFPU init holds at zero, and an INCRWC after it
// steps DEST: ADDR_MOD_6 belongs to the FPU ops (matmul) a kernel may run between this init and its call.
template <class Config>
inline void init_reciprocal_complement()
{
    sfpu_reciprocal_init<false>();
#if !defined(DISABLE_SFPLOADMACRO)
    // The body's load macro: its two instruction templates, its sequence and its misc settings.
    TTI_SFPSETSGN(0, p_sfpu::LREG2, 15, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, Config::kMacroSequenceBits >> 16);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, Config::kMacroSequenceBits & 0xffffu);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
#endif
}

template <class Config, int Iterations = 32>
inline void calculate_reciprocal_complement()
{
    static_assert(Iterations == 32);
    // Degree four leaves LREG5 out of the replayed body, so it can carry the NaN rows.
    static_assert(Config::kDegree == 4);
    static_assert(Config::kBodySlots == reciprocal_complement_body_slots(Config::kDegree));
    // The body overwrites each row with its result and DEST past this tile holds the
    // caller's other tiles, so whether row d is NaN is kept as bit 31 - d of LREG5.
    sfpi::vInt nan_rows = 0;
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d)
    {
        sfpi::vFloat raw = sfpi::dst_reg[d];
        nan_rows <<= 1;
        v_if (sfpi::as<sfpi::vInt>(sfpi::setsgn(raw, 0)) > 0x7f800000)
        {
            nan_rows |= 1;
        }
        v_endif;
    }
    sfpi::l_reg[sfpi::LRegs::LReg5] = nan_rows;
    // The rungs read c2 and c1 from LREG7 and LREG6.
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[2] >> 16);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[2] & 0xffffu);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, Config::kComplementBits >> 16);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, Config::kComplementBits & 0xffffu);
#if defined(DISABLE_SFPLOADMACRO)
    // The same rows without the load macro or the replay buffer: identical results.
#pragma GCC unroll 1
    for (int d = 0; d < 32; ++d)
    {
        ttpoly::atan::reciprocal_complement_body<Config, ADDR_MOD_7, ADDR_MOD_7, false, true>();
        TTI_INCRWC(0, 2, 0, 0);
    }
#else
    load_replay_buf<Exec>(
        0,
        Config::kBodySlots + ttpoly::atan::reciprocal_complement_held_slots,
        [&]
        {
            ttpoly::atan::reciprocal_complement_body<Config, ADDR_MOD_7, ADDR_MOD_7, true, true>();
            TTI_INCRWC(0, 2, 0, 0);
        });
#pragma GCC unroll 8
    for (int d = 1; d < 32; ++d)
    {
        TTI_REPLAY(0, Config::kBodySlots + ttpoly::atan::reciprocal_complement_held_slots, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
    // The body stored every row rounded to BF16, so only a NaN row is rewritten,
    // to +Inf for either sign. The traversal restarts at row 0, independently of caller cleanup.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    sfpi::vInt nan_flags = sfpi::l_reg[sfpi::LRegs::LReg5];
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d)
    {
        v_if (nan_flags < 0)
        {
            sfpi::dst_reg[d] = ttpoly::atan::target_raw_terminal_value<1>(sfpi::vFloat(0.0f));
        }
        v_endif;
        nan_flags <<= 1;
    }
}
} // namespace ckernel::sfpu::bf16::atan
