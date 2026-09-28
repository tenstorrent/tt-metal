// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sfpi.h"

// Compensated BF16 recurrent state (COMPENSATED and LOW_PRECISION).
//
// The online-softmax numerator O and denominator l are carried between K chunks as BF16 pairs,
// value = hi + lo, with |lo| <= ulp(hi)/2. Each K chunk folds its BF16 contribution `chunk` into the
// state with the online-softmax correction c = exp(scale * (m_old - m_new)):
//
//     root  = hi + lo                  FP32; exact, since hi and lo together fit in 16 significant bits
//     total = root * c + chunk         FP32 (one SFPMAD)
//     hi'   = RNE_bf16(total)
//     lo'   = RNE_bf16(total - hi')    total - hi' is exact in FP32
//
// This is round-and-split, not a full TwoSum: the state keeps about 16 significant bits instead of 8, so
// the relative error added per K chunk drops from about 2^-9 to about 2^-17. It fixes the swamping of a
// long BF16 recurrence. The BF16 PV products within a chunk, and the BF16 correction c itself, are
// unchanged.
//
// The routines run from the SFPU replay buffer with SFPLOADMACRO templates (0/1: FP32 -> BF16 RNE of
// L0/L3; 2/3: add a loaded low part to a held high part; results are stored two instructions later at the
// load's captured DST address). Replay slots 0-14 hold the numerator block program and 15-30 the
// denominator program; init_group2_identity_replay re-records slots 0-17, so callers re-run the
// matching init before each use. DST offsets are in SFPU rows; one tile is 64 rows apart.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Denominator update for two rows: DST [hi0, lo0, chunk0, hi1, lo1, chunk1, c0, c1]. Unlike the
// numerator, each row has its own correction (L6, L7). Records replay slots 15-30 beside the numerator's
// slots 0-14; both use the templates from init_sdpa_compensated_block_macros.
inline void init_sdpa_compensated_sum_replay() {
    TTI_REPLAY(15, 16, 0, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 448);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 192);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 128);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    TTI_SFPMAD(2, 7, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 192);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 256);
}

inline void calculate_sdpa_zero_sum() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg++;
    }
}

// Numerator update for one row of two column tiles: DST [hi0, hi1, lo0, lo1, chunk0, chunk1, c], with the
// column-broadcast correction c shared by both tiles. Configures the macro templates and records slots 0-14:
//   L6 = c; L1 = hi0 + lo0; L2 = hi1 + lo1; L0 = L1 * L6 + chunk0; L3 = L2 * L6 + chunk1
//   store RNE(L0), RNE(L3) as hi'; L0 -= hi0'; L3 -= hi1'; store RNE(L0), RNE(L3) as lo'
inline void init_sdpa_compensated_block_macros() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    // Templates 0/1 round fixed VC=L0/L3. Override VB with the load's
    // register, leaving VC intact; store that rounded register two instructions
    // later at the load's captured DST address.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 3, 3, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0x1384);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, 0x8, 0x1385);
    TTI_SFPCONFIG(0, 5, 0);
    // Templates 2/3 add the loaded low component to the fixed high component.
    TTI_SFPADD(10, 0, 1, 14, 0);
    TTI_SFPADD(10, 3, 2, 15, 0);
    TTI_SFPCONFIG(0x600, 6, 1);
    TTI_SFPCONFIG(0x700, 7, 1);
    TTI_SFPCONFIG(0xF00, 8, 1);  // BF16 stores; delays count SFPU instructions.

    // Record only: no DST access occurs until a subsequent tile_regs_wait.
    // DST holds (hi0, hi1, lo0, lo1, chunk0, chunk1, correction).
    // L0/L3 retain full sums; L1/L2 hold rounded results; L4/L5 are chunks.
    // Keep macro-load destinations below L4: VDHi also encodes address bit 0.
    TTI_REPLAY(0, 15, 0, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 192);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    TTI_SFPMAD(2, 6, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 64);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 192);
}

// Apply the recorded numerator (or, with separate_corrections, denominator) update to a full tile pair.
// Every second pass replays the program without its first instruction, reusing the correction already in
// L6[/L7] instead of reloading it.
template <int pairs, bool separate_corrections = false>
inline void calculate_sdpa_compensated_reuse() {
    static_assert(pairs == 2);
    {
#pragma GCC unroll 8
        for (int i = 0; i < 32; i += 2) {
            TTI_REPLAY(separate_corrections ? 15 : 0, separate_corrections ? 16 : 15, 0, 0);
            // Skip only SFPLOAD(s) of L6[/L7]. Keep every arithmetic and store
            // instruction and the original automatic DST increment intact.
            TTI_REPLAY(separate_corrections ? 17 : 1, 14, 0, 0);
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

// identity: the row maximum did not change, so c = 1. Load exactly 1.0 as the correction and replay the
// update without its correction load; the result equals the general update with a 1.0 correction tile.
template <int pairs, bool separate_corrections = false>
inline void calculate_sdpa_identity_state(bool identity) {
    if (!identity) {
        calculate_sdpa_compensated_reuse<pairs, separate_corrections>();
        return;
    }
    static_assert(pairs == 2, "The replay programs update two tiles per pass");
    // Called only after the current DST wait. Exact constant substitutes for
    // the original BF16 correction loads, not for any MAD or state arithmetic.
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_FLOATB, 0x3f80);
    if constexpr (separate_corrections) {
        TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_FLOATB, 0x3f80);
    }
#pragma GCC unroll 8
    for (int i = 0; i < 32; ++i) {
        TTI_REPLAY(separate_corrections ? 17 : 1, 14, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Two-chunk folds (compensated_group.hpp). When consecutive K chunks leave the row maximum unchanged, the
// first chunk's BF16 PV output is kept as a `local` term and both chunks fold into the state at once,
// halving the number of round-and-split steps:
//   identity fold (c = 1):  total = (hi + lo) + (local + chunk)
//   changed fold:           total = (hi + lo + local) * c + chunk
// followed by hi' = RNE(total), lo' = RNE(total - hi'). Odd: no local term (the group had one chunk);
// the local slot is zeroed for the next group.
//
// Identity fold for two rows. DST: hi[0:2], lo[2:4], local[4:6], chunk[6:8] (tiles).
template <bool Odd>
inline void calculate_group2_identity_fold_impl() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
#pragma GCC unroll 4
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat old_high0 = sfpi::dst_reg[0];
        sfpi::vFloat old_low0 = sfpi::dst_reg[64];
        sfpi::vFloat root0 = old_high0 + old_low0;
        sfpi::vFloat old_high1 = sfpi::dst_reg[32];
        sfpi::vFloat old_low1 = sfpi::dst_reg[96];
        sfpi::vFloat root1 = old_high1 + old_low1;
        sfpi::vFloat local0 = 0.0f;
        if constexpr (!Odd) {
            local0 = sfpi::dst_reg[128];
        }
        sfpi::vFloat chunk0 = sfpi::dst_reg[192];
        sfpi::vFloat group0 = local0 + chunk0;
        sfpi::vFloat local1 = 0.0f;
        if constexpr (!Odd) {
            local1 = sfpi::dst_reg[160];
        }
        sfpi::vFloat chunk1 = sfpi::dst_reg[224];
        sfpi::vFloat group1 = local1 + chunk1;
        sfpi::vFloat total0 = root0 + group0;
        sfpi::vFloat total1 = root1 + group1;
        sfpi::vFloat high0 = sfpi::convert<sfpi::vFloat16b>(total0, sfpi::RoundMode::Nearest);
        sfpi::vFloat high1 = sfpi::convert<sfpi::vFloat16b>(total1, sfpi::RoundMode::Nearest);
        sfpi::vFloat low0 = sfpi::convert<sfpi::vFloat16b>(total0 - high0, sfpi::RoundMode::Nearest);
        sfpi::vFloat low1 = sfpi::convert<sfpi::vFloat16b>(total1 - high1, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = high0;
        sfpi::dst_reg[32] = high1;
        sfpi::dst_reg[64] = low0;
        sfpi::dst_reg[96] = low1;
        if constexpr (Odd) {
            sfpi::dst_reg[128] = 0.0f;
            sfpi::dst_reg[160] = 0.0f;
        }
        sfpi::dst_reg++;
    }
}

// Changed-maximum fold for one row. DST: hi, lo, local, chunk, column-broadcast correction c.
template <bool Odd>
inline void calculate_group2_changed_fold_impl() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
#pragma GCC unroll 4
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat old_high = sfpi::dst_reg[0];
        sfpi::vFloat old_low = sfpi::dst_reg[32];
        sfpi::vFloat root = old_high + old_low;
        sfpi::vFloat local = 0.0f;
        if constexpr (!Odd) {
            local = sfpi::dst_reg[64];
        }
        sfpi::vFloat combined = root + local;
        sfpi::vFloat correction = sfpi::dst_reg[128];
        sfpi::vFloat chunk = sfpi::dst_reg[96];
        sfpi::vFloat total = combined * correction + chunk;
        sfpi::vFloat high = sfpi::convert<sfpi::vFloat16b>(total, sfpi::RoundMode::Nearest);
        sfpi::vFloat low = sfpi::convert<sfpi::vFloat16b>(total - high, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = high;
        sfpi::dst_reg[32] = low;
        if constexpr (Odd) {
            sfpi::dst_reg[64] = 0.0f;
        }
        sfpi::dst_reg++;
    }
}
inline void calculate_group2_identity_odd_fold() { calculate_group2_identity_fold_impl<true>(); }
inline void calculate_group2_changed_fold() { calculate_group2_changed_fold_impl<false>(); }
inline void calculate_group2_changed_odd_fold() { calculate_group2_changed_fold_impl<true>(); }

// Replay form of the two-row identity fold (slots 0-17; overwrites the numerator block program).
inline void init_group2_identity_replay() {
    // Same root-add and BF16 round/store templates as the numerator block update.
    init_sdpa_compensated_block_macros();
    // DST: hi0,hi1,lo0,lo1,local0,local1,chunk0,chunk1.
    // Macro outputs root0/root1=L1/L2; full totals=L0/L3.
    TTI_REPLAY(0, 18, 0, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 192);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 448);
    TTI_SFPADD(10, 4, 6, 4, 0);
    TTI_SFPADD(10, 5, 7, 5, 0);
    TTI_SFPADD(10, 1, 4, 0, 0);
    TTI_SFPADD(10, 2, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 64);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 192);
}
inline void calculate_group2_identity_replay() {
#pragma GCC unroll 8
    for (int i = 0; i < 32; ++i) {
        TTI_REPLAY(0, 18, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu
#endif
