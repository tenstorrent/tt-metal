// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "sfpi.h"
namespace ckernel::sfpu {
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
}  // namespace ckernel::sfpu
#endif
