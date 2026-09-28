// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_reduce.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_binary_bcast.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

// sfpu/ckernel_sfpu_binary_bcast.h (included above) used to define these; #49682 moved that header to
// LTILEID-derived lane masks and dropped them. Values carried over verbatim. SFPCONFIG target index used
// with an immediate mask to force LReg[11] = 1.0 on specific SFPU instances (= specific "SFPU columns"
// within each 8-lane group) and 0.0 on the others. Bit N of the mask corresponds to SFPU instance N; the
// low 8 bits control the 8 SFPU columns.
constexpr std::uint32_t SFPCONFIG_TARGET_LREG11  = 11;
constexpr std::uint32_t SFPCONFIG_MOD_SET_LREG11 = 8;

template <bool is_fp32_dest_acc_en>
inline void _init_softmax_k_()
{
    sfpu::exp_init<false, 0x3F800000, true, is_fp32_dest_acc_en>();
}

// For odd k, the final valid even lane predicates its paired odd tail lane.
// Clear that extra exponential in LREG5 (the odd-column exponentials, loaded by the caller) before it
// contributes to the row sum. Clobbers LREG0; LREG11 is restored to -1.0 on exit.
template <int k>
inline void _zero_paired_odd_tail_lane_()
{
    if constexpr ((k & 1) && k < 16)
    {
        constexpr std::uint32_t all_instances_mask = 0x5555;
        constexpr std::uint32_t tail_instance_mask = 1u << (k - 1);

        // Mark only the SFPU instance containing odd tail lane k.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
        TTI_SFPCONFIG(all_instances_mask, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
        TTI_SFPCONFIG(tail_instance_mask, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);

        TTI_SFPSETCC(0, p_sfpu::LREG11, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // Restore LREG11 to its hardware-default -1.0 value.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xBF80);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
        TTI_SFPCONFIG(all_instances_mask, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);
    }
}

template <int k, bool is_fp32_dest_acc_en>
inline void _softmax_k_()
{
    // LREG0 = x - max(x)
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
    TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 1);

    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 8);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);

    // //LREG0 = exp(x - max(x))
    sfpu::calculate_exponential<
        false,
        is_fp32_dest_acc_en,
        false, // scaling
        2,     // iterations
        true   // clamp negatives
        >();

    TTI_SFPENCC(0, 0, 0, 0);
    math::clear_dst_reg_addr();

    // LREG4/LREG5 = exp(x - max(x)) for the even/odd columns. They stay live until the final normalize, so the
    // exponentials are read back from DEST exactly once (the odd-k tail lane is zeroed in LREG5 itself; the
    // normalize overwrites that DEST word with 0 * 1/sum = 0, as the old store/reload round trip did).
    TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
    _zero_paired_odd_tail_lane_<k>();

    // LREG0 = sum(exp(x - max(x))) over the 16 columns, in every one of the 8 lanes of each row.
    //
    // Column pair first (a_i = even_i + odd_i), then a cyclic rotate-and-add over the 8 lanes of each row with the
    // rotate distance DEcreasing 4, 2, 1. With that order every lane ends up with the same association,
    //   ((a[i] + a[i^4]) + (a[i^2] + a[i^6])) + ((a[i^1] + a[i^5]) + (a[i^3] + a[i^7]))   (^ = xor)
    // (after rot4 each value is 4-periodic, after rot2 2-periodic), so all 8 lanes hold the identical sum and no
    // column-0 gather / re-broadcast is needed. It is the same tree as the column-0 result of
    // horizontal_reduce<false>() that this replaces (lane 0 even keeps its operand roles; the other lanes see
    // a + b as b + a at some levels, which SFPADD rounds identically), so the sum is bit-identical to it.
    //
    // SFPSHFT2 VD != VC rotates straight out of the accumulator (no SFPMOV copy). Blackhole auto-stalls the
    // instruction after a SUBVEC_SHFLROR1, so SHFT2 -> SHFT2 / SHFT2 -> SFPADD need no padding, but its scoreboard
    // does not see the SHFLROR1 read (tt-isa-documentation SFPMAD.md, "hardware bugs"): each SFPADD -> SFPSHFT2
    // hand-off needs the SFPNOP.
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    // Rotate by 4.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    // Rotate by 2.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    // Rotate by 1.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);

    // LREG0 = 1 / sum(exp(x - max(x))).
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    // Normalize both column groups and write the completed softmax to row 0.
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
}

} // namespace sfpu
} // namespace ckernel
