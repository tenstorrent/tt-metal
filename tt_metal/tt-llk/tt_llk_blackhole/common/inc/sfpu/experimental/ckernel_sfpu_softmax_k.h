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
// within each 8-lane group) and 0.0 on the others. Bit 2*N of the mask selects SFPU instance N
// (SFPCONFIG.md, MOD1_IMM16_IS_LANE_MASK), so 0x5555 selects all eight.
constexpr std::uint32_t SFPCONFIG_TARGET_LREG11  = 11;
constexpr std::uint32_t SFPCONFIG_MOD_SET_LREG11 = 8;
constexpr std::uint32_t SFPCONFIG_ALL_INSTANCES  = 0x5555;

// SFPCONFIG lane mask selecting SFPU instances first_instance .. 7 (0 for first_instance == 8).
constexpr std::uint32_t _instances_from_(int first_instance)
{
    return SFPCONFIG_ALL_INSTANCES & ~((1u << (2 * first_instance)) - 1u);
}

template <bool is_fp32_dest_acc_en>
inline void _init_softmax_k_()
{
    sfpu::exp_init<false, 0x3F800000, true, is_fp32_dest_acc_en>();
}

// Zero the lanes of one column block of the row band (addr 0 = even columns, addr 2 = odd columns) that
// belong to the SFPU instances whose LREG11 is non-zero. Leaves every lane enabled.
template <std::uint32_t addr>
inline void _zero_marked_instances_()
{
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, addr);
    TTI_SFPSETCC(0, p_sfpu::LREG11, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, addr);
    TTI_SFPENCC(0, 0, 0, 0);
}

// Zero the padding columns (>= k) that still hold a value after the exponential, by SFPU instance index.
//
// Lane L of an SFPLOAD/SFPSTORE at address 0 is DEST column 2*(L%8) and at address 2 column 2*(L%8)+1
// (SFPLOAD.md), for a 16-bit and a 32-bit DEST alike, so SFPU instance j = L%8 owns the column pair
// (2j, 2j+1): the padding columns are the instances >= ceil(k/2) in the even block and >= floor(k/2) in the
// odd block. Which of them need clearing depends on the exponential that ran:
//  - 16-bit DEST: _sfpu_exp_21f_bf16_tti_ is pure TTI code and keeps the SFPGT lane predicate, so the
//    padding columns stay 0 except, for odd k, the odd tail column k, whose even partner k-1 is a valid
//    lane and therefore enabled their shared instance (k-1)/2 (mask bit k-1).
//  - 32-bit DEST: _sfpu_exp_fp32_accurate_ is sfpi code whose top-level v_endif re-enables every lane
//    (SFPENCC) before its store, so the predicate is gone: the first store writes all 32 lanes of the even
//    block and the odd block is exponentiated unpredicated (exp(0) = 1 in every padding lane). Every
//    padding column of both blocks has to be re-zeroed.
template <int k, bool is_fp32_dest_acc_en>
inline void _zero_padding_columns_()
{
    // k is the number of valid columns of the 16-wide row; the 16-bit tail mask below shifts by k - 1 into
    // SFPCONFIG's 16-bit immediate, so anything past 16 would spill into the opcode.
    static_assert(k >= 1 && k <= 16, "softmax_k: k must be in 1..16");
    constexpr std::uint32_t even_block_mask = is_fp32_dest_acc_en ? _instances_from_((k + 1) / 2 /*first_instance*/) : 0u;
    constexpr std::uint32_t odd_block_mask  = is_fp32_dest_acc_en ? _instances_from_(k / 2 /*first_instance*/) : ((k & 1) ? (1u << (k - 1)) : 0u);
    static_assert((odd_block_mask & even_block_mask) == even_block_mask, "the odd block clears a superset of the even block's instances");

    if constexpr (even_block_mask != 0 || odd_block_mask != 0)
    {
        // LREG11 = 1.0 on the instances to clear, 0.0 elsewhere (SFPCONFIG takes the value from LREG0).
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
        TTI_SFPCONFIG(SFPCONFIG_ALL_INSTANCES, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);

        if constexpr (even_block_mask != 0)
        {
            TTI_SFPCONFIG(even_block_mask, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);
            _zero_marked_instances_<0 /*addr*/>();
        }
        if constexpr (odd_block_mask != 0)
        {
            if constexpr (odd_block_mask != even_block_mask)
            {
                // Superset of even_block_mask and LREG0 still holds 1.0: marking the extra instances is enough.
                TTI_SFPCONFIG(odd_block_mask, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);
            }
            _zero_marked_instances_<2 /*addr*/>();
        }

        _restore_lconst_neg1_();
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
    _zero_padding_columns_<k, is_fp32_dest_acc_en>();

    // LREG0 = sum(exp(x - max(x)))
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, 0, 0);
    TTI_SFPNOP;
    sfpu::horizontal_reduce<false>();

    // Broadcast the column-0 sum across all eight SFPU columns.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG7, 0);
    _build_lane_mask_col0_();
    TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, LREG_MASK, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    // LREG0 = 1 / sum(exp(x - max(x))).
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    // Normalize both column groups and write the completed softmax to row 0.
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 2);
}

} // namespace sfpu
} // namespace ckernel
