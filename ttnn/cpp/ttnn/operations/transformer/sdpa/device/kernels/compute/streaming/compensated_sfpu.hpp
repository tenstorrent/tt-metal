// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sfpi.h"

// High/low BF16 state arithmetic. Keep replay slots, DST layouts and rounding
// order explicit: these routines share the SFPU macro/replay resource.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Same update as the paired numerator, but each denominator row owns its
// correction. Eight BF16 tiles fill one half-DST. Keep this 16-instruction
// program beside the numerator's 15 entries without reconfiguring macros.
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

template <int pairs, bool separate_corrections = false>
inline void calculate_sdpa_identity_state(bool identity) {
    if (!identity) {
        calculate_sdpa_compensated_reuse<pairs, separate_corrections>();
        return;
    }
    static_assert(pairs == 2, "Fixed two-state numerator/denominator prototype");
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

// Two output states: DST hi[0:2], lo[2:4], local[4:6], chunk[6:8].
// No intermediate BF16 local sum is created at this fold.
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

// One output state: hi,lo,local,chunk,canonical COL-broadcast correction.
// The entire protected+local state is corrected before adding current PV.
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
inline void calculate_group2_identity_fold() { calculate_group2_identity_fold_impl<false>(); }
inline void calculate_group2_identity_odd_fold() { calculate_group2_identity_fold_impl<true>(); }
inline void calculate_group2_changed_fold() { calculate_group2_changed_fold_impl<false>(); }
inline void calculate_group2_changed_odd_fold() { calculate_group2_changed_fold_impl<true>(); }

inline void init_group2_identity_replay() {
    // Same root-add and BF16 round/store templates as the frozen compensation.
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
