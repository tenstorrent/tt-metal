// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// SFPLOAD at DST offsets d and d+2 reads even and odd columns of the same
// four physical rows. A COL-broadcast correction is lane-wise identical for
// this pair. Reuse only that vector, not correction state across DST halves.
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
template <int pairs, bool separate_corrections = false>
inline void calculate_sdpa_compensated_reuse() {
    if constexpr (pairs == 2) {
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
    } else {
        calculate_sdpa_compensated_state<pairs, separate_corrections>();
    }
}
}  // namespace ckernel::sfpu
#endif
