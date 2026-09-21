// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
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
}  // namespace ckernel::sfpu
#endif
