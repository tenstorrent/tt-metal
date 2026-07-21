// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

constexpr std::uint32_t dst_tile_offset = 64; // 1 tile x 64 rows per tile
constexpr std::uint32_t scores_offset   = 0;
constexpr std::uint32_t indices_offset  = scores_offset + dst_tile_offset;
constexpr std::uint32_t bias_offset     = indices_offset + dst_tile_offset;
constexpr std::uint32_t interm_offset   = bias_offset + dst_tile_offset;

template <bool is_fp32_dest_acc_en, std::uint32_t offset = 0>
inline void bitonic_topk_load16_concat_indices_single_face()
{
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 12 + offset);

    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be false");
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 12 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 12 + offset);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_even_cols_concatted_indices_single_face()
{
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_split_indices_single_face()
{
    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be false");
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_concatted_indices_single_face()
{
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph0_st1_to_1_single_face()
{
    if constexpr (start_transpose)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    if constexpr (end_transpose)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph1_st2_to_1_single_face()
{
    if constexpr (start_transpose)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    if constexpr (end_transpose)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool end_transpose, bool bitonic = true>
inline void bitonic_topk_ph2_st3_to_1_single_face()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    if constexpr (bitonic)
    {
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);

    constexpr int swap_mode = bitonic ? p_sfpswap::ROWS_01_MAX : p_sfpswap::ALL_ROWS_MAX;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, swap_mode);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, swap_mode);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, swap_mode);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, swap_mode);

    if constexpr (end_transpose)
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool dir, bool end_transpose>
inline void bitonic_top8_ph3_st4_to_1()
{
    // TODO: Use replay buffer for these instructions
    if constexpr (dir == (bool)SortDir::ArgMax)
    {
        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 1
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (end_transpose)
        {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
    else
    {
        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        if constexpr (end_transpose)
        {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir>
inline void bitonic_top8_ph0_to_ph3()
{
    // Phase 0
    {
        constexpr bool start_transpose = true;
        constexpr bool end_transpose   = false;
        bitonic_topk_ph0_st1_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 1
    {
        constexpr bool start_transpose = false;
        constexpr bool end_transpose   = true;
        // Odd Columns
        bitonic_topk_ph1_st2_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 2
    {
        constexpr bool end_transpose = true;
        // Even Columns
        bitonic_topk_ph2_st3_to_1_single_face<end_transpose>();
    }
    // Modified Phase 3 for top8
    {
        constexpr bool end_transpose = true;
        bitonic_top8_ph3_st4_to_1<idir, end_transpose>();
    }
}

inline void reverse_sort_order()
{
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_sum_top2()
{
    constexpr bool idir = false; // Sort descending order

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPCONFIG(0x4, 0xF, 1);

    // Phase 0-3 Even Columns
    bitonic_topk_load16_concat_indices_single_face<is_fp32_dest_acc_en, 0>();
    bitonic_top8_ph0_to_ph3<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir>();
    bitonic_topk_store8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Phase 0-3 Odd Columns
    bitonic_topk_load16_concat_indices_single_face<is_fp32_dest_acc_en, 2>();
    bitonic_top8_ph0_to_ph3<APPROXIMATION_MODE, is_fp32_dest_acc_en, !idir>();

    // Instead of a full phase 4, we rerun phase 3 since we are only comparing top8 values
    bitonic_topk_load8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();
    bitonic_top8_ph3_st4_to_1<idir, true>();
    bitonic_topk_store8_even_cols_split_indices_single_face<is_fp32_dest_acc_en>();

    // Sum top2
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Replicate the top2 sum down the column
    // This is needed for the subsequent top4 sort
    // TODO: Evaluate compared to broadcast using MOVB2D in FPU
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG14, p_sfpu::LREG0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_sort_top4_groups()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // Sort top4
    // Load the top2 sums and concat indices
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, interm_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4);

    // Load the top2 sums (again) and bias scores
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_3, bias_offset + 4);

    // Sort 8 groups (not bitonic)
    bitonic_topk_ph0_st1_to_1_single_face<true, false>();
    bitonic_topk_ph1_st2_to_1_single_face<false, true>();
    bitonic_topk_ph2_st3_to_1_single_face<true, false>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_3, bias_offset + 0);
}

// P2 building block: merge the 4 sorted-8 runs sitting in the canonical column-major layout
// (bias/indices/scores at offsets +0/+2/+4/+6) into the RAW top-8 (no normalization).
// Extracted verbatim from the merge portion of _generalized_moe_gate_top8. After it returns,
// the top-8 expert indices are in indices+0/+4 and the (un-normalized) scores in scores+0/+4.
// Internally does "4 runs -> 2 -> 1" via two bitonic_top8_ph3_st4_to_1 stages; each stage's two
// inputs are made bitonic by reverse_sort_order, exactly as the original top8 did.
template <bool is_fp32_dest_acc_en, std::uint32_t base = 0>
inline void _gmg_merge4_runs_raw()
{
    constexpr bool idir = false; // Sort descending order

    // Combine and sort 4 groups of 8 values to 2 groups of 8 values
    // Even Columns sorted Top8 in LREG0 and LREG1. `base` selects which 4-run block to read
    // (0 = groups in cols +0/+2/+4/+6; 8 = the next block at +8/+10/+12/+14, if step1 filled it).
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Reverse order of sort for top8 values in odd columns
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + base + 6);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + base + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + base + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + base + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + base + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + base + 2);
    reverse_sort_order();

    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + base + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + base + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + base + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + base + 4);
    bitonic_top8_ph3_st4_to_1<idir, true>();

    bitonic_topk_store8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Move and reverse the other column of 8 values
    // Disable index tracking while we shift values
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    // There is a hardware bug that affects operations on LREG4-7 while this is enabled
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    // Re-enable index tracking
    TTI_SFPCONFIG(0x4, 0xF, 1);
    reverse_sort_order();
    bitonic_topk_load8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Step 4 Only, we need top8 but it doesn't have to be sorted
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_top8(std::uint32_t eps, std::uint32_t scale)
{
    // Merge the 4 runs into the raw top-8 (idx -> indices+0/+4, score -> scores+0/+4).
    _gmg_merge4_runs_raw<is_fp32_dest_acc_en, 0>();

    // ---- normalization tail: 1/(sum+eps)*scale, broadcast-multiply the top-8 scores ----
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);

    // Reduce the top8 values to 1 value
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    // Calculate 1 / (sum + eps) * scale
    // Store the value in lreg0 and reload later since the following instructions overwrite it
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 0);
    // Technically we only need to reprogram a single constant here, instead of all 3 used by reciprocal init
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 0);
    sfpi::vFloat l0                 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value          = Converter::as_float(eps);
    l0                              = l0 + eps_value;
    l0                              = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value        = Converter::as_float(scale);
    l0                              = l0 * scale_value;
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    TTI_SFPNOP;

    // Broadcast to all rows and multiply by the top8 values
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);
}

/**
 * @brief Merge 4 sorted-descending top-8 runs at DEST columns {read_base, +2, +4, +6} into a single top-8
 *        and store it (bias value + LO16 index + HI16 score) at columns {store_lo, store_hi}.
 *
 * This is the existing top8's 4-column merge (stage 1 + stage 2 of _generalized_moe_gate_top8) with the
 * initial load offsets and final store offsets parameterized; stage 2 is base-independent. SFPU (MATH).
 *
 * @tparam is_fp32_dest_acc_en: DEST accumulation mode; must be false — the idx|score concat relies on 16-bit
 *         (bf16) DEST packing (LO16 index / HI16 score).
 * @tparam read_base: DEST column base of the 4 input runs (loaded at read_base + {0,2,4,6}).
 * @tparam store_lo: DEST column for the merged run's low half (bias/idx/score).
 * @tparam store_hi: DEST column for the merged run's high half.
 * @note Resets the Dst RWC counter at entry (TTI_SETRWC SET_D): a preceding FPU MOP (e.g. copy4rows) leaves
 *       it advanced by +64/tile, which would bias the SFPLOAD offsets — so run this after any such MOP.
 *       Reads bias(mode 0)/indices(LO16)/scores(HI16) at the read columns and writes the same convention at
 *       the store columns; clobbers LREG0-7. @ref _gmg_merge16_to_run is the full-16 (two-run) variant.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t read_base, std::uint32_t store_lo, std::uint32_t store_hi>
inline void _gmg_merge4_top8()
{
    constexpr bool idir = false; // descending
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Odd columns (read_base+6, read_base+2), reversed to form a bitonic sequence.
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + read_base + 6);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + read_base + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + read_base + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + read_base + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + read_base + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + read_base + 2);
    reverse_sort_order();

    // Even columns (read_base+0, read_base+4).
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + read_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + read_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + read_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + read_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + read_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + read_base + 4);
    bitonic_top8_ph3_st4_to_1<idir, true>();
    bitonic_topk_store8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Stage 2: shift in "the other column of 8", merge (base-independent — identical to
    // _generalized_moe_gate_top8 lines for the second merge stage).
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPCONFIG(0x4, 0xF, 1);
    reverse_sort_order();
    bitonic_topk_load8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    // Result: merged top-8 — bias values in LREG0/LREG1, concat(idx|score) in LREG4/LREG5.
    // Store as a re-mergeable run at columns {store_lo, store_hi} so the existing top8 can
    // consume cols {0,2,4,6} for the final merge.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + store_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + store_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + store_hi);
}

/**
 * @brief Copy a stored top-8 run (bias + concat idx|score) from DEST column pair {from_lo,from_hi} to
 *        {to_lo,to_hi}, matching _gmg_merge4_top8's store convention (bias mode 0; idx LO16; score HI16).
 *
 * Used to relocate a saved run (e.g. topA parked at safe rows 8-15) into the final merge slot. SFPU (MATH).
 *
 * @tparam from_lo: source DEST column for the run's low half.
 * @tparam from_hi: source DEST column for the run's high half.
 * @tparam to_lo: destination DEST column for the low half.
 * @tparam to_hi: destination DEST column for the high half.
 * @note Resets the Dst RWC counter at entry (TTI_SETRWC SET_D): a preceding FPU MOP (e.g. copy4rows) leaves
 *       it advanced, which would bias the SFPLOAD/SFPSTORE offsets — run this after any such MOP. Clobbers
 *       LREG0/1/4/5.
 */
template <std::uint32_t from_lo, std::uint32_t from_hi, std::uint32_t to_lo, std::uint32_t to_hi>
inline void _gmg_copy_topk_run()
{
    // Reset the Dst RWC counter (a preceding FPU MOP — e.g. copy4rows — leaves it advanced by +64/tile;
    // without this the SFPLOAD offsets below are biased by that leftover and read the wrong rows).
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + from_lo);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + from_hi);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + from_lo);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + from_hi);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + from_lo);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + from_hi);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + to_lo);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + to_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + to_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + to_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + to_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + to_hi);
}

// Multi-block combine: place ONE field of a run from the intermediate region {src_lo,src_hi} into its home
// region (scores/indices/bias) {dst_lo,dst_hi}. The field-tile was unpacked (copy_tile) into intermediate
// from the L1 run stash; this SFPU copy is row-selective so it writes only {dst_lo,dst_hi}, leaving
// the other block's run (sitting at the complementary rows) intact. mode/region match the store
// convention of _gmg_copy_topk_run/_gmg_merge4_top8 (field 0=bias mode 0; 1=idx LO16; 2=score HI16).
template <std::uint32_t field, std::uint32_t src_lo, std::uint32_t src_hi, std::uint32_t dst_lo, std::uint32_t dst_hi>
inline void _gmg_place_field_from_interm()
{
    constexpr std::uint32_t mode = (field == 0) ? 0 : (field == 1) ? (std::uint32_t)InstrModLoadStore::LO16_ONLY : (std::uint32_t)InstrModLoadStore::HI16_ONLY;
    constexpr std::uint32_t region = (field == 0) ? bias_offset : (field == 1) ? indices_offset : scores_offset;
    // Reset Dst RWC: the copy_tile (FPU) that filled intermediate leaves it advanced; without this the
    // SFPLOAD/SFPSTORE offsets below are biased and hit the wrong rows.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPLOAD(p_sfpu::LREG0, mode, ADDR_MOD_3, interm_offset + src_lo);
    TTI_SFPLOAD(p_sfpu::LREG1, mode, ADDR_MOD_3, interm_offset + src_hi);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, mode, ADDR_MOD_3, region + dst_lo);
    TTI_SFPSTORE(p_sfpu::LREG1, mode, ADDR_MOD_3, region + dst_hi);
}

// Ungrouped finalize: combine topA at {0,2} (rows 0-3) and topB at {4,6} (rows 4-7) into the global
// top-8, then normalize. The two sorted-8 runs from merge4_top8 are in a lane layout that the 2-run
// bitonic-merge primitives (ph3 / merge4_runs_raw) mis-handle. Instead, load all 16 candidates as an
// UNSORTED 16-vector (topA -> LREG0/1, topB -> LREG2/3; idx|score in LREG4-7) and run the FULL bitonic
// sort. ph0_to_ph3 fully sorts 16 arbitrary values -> the global top-8, independent of run orientation.
// merge16 CORE: load the 16 candidates at {0,2,4,6} (two sorted-8 runs at {0,2} and {4,6}) + concat
// idx(LO16)|score(HI16), and FULL-sort -> global top-8 in LREG0/1 (bias) + LREG4/5 (concat idx|score).
// Shared by the per-block merge (topA+topB) AND every level of the multi-block combine tree.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _gmg_merge16_core()
{
    constexpr bool idir = false; // descending -> top-8
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPCONFIG(0x4, 0xF, 1); // enable index tracking (idx|score in LREG4-7 follow the bias swaps)

    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 6);

    bitonic_top8_ph0_to_ph3<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir>();
}

/**
 * @brief Full-sort the 16 candidates (two sorted-8 runs at DEST columns {0,2}+{4,6}) into the global top-8
 *        and store it as a re-mergeable run at {store_lo, store_hi}; no normalize.
 *
 * Same store convention as _gmg_merge4_top8 (bias mode 0; idx LO16; score HI16). Used for a block's top-8
 * and for combine-tree intermediates, so a later merge16 (reading {0,2}+{4,6}) can consume the run. SFPU (MATH).
 *
 * @tparam APPROXIMATION_MODE: fast-approx mode forwarded to the bitonic sort.
 * @tparam is_fp32_dest_acc_en: DEST accumulation mode; must be false.
 * @tparam store_lo: DEST column for the run's low half.
 * @tparam store_hi: DEST column for the run's high half.
 * @tparam idx_offset: per-block expert-id base (b*256) added to the run's indices to make them GLOBAL; 0 = none.
 * @note Calls @ref _gmg_merge16_core, which resets the Dst RWC counter and enables SFPU index tracking, and
 *       reads the two input runs at columns {0,2} and {4,6}; clobbers LREG0-7 (and LREG14 via the core). When
 *       idx_offset != 0 it adds it to the idx (LO16) of the concat with a raw SFPIADD (idx + offset <= 511
 *       never carries into the score HI16). Produces a RUN, not a normalized output — @ref
 *       _generalized_moe_gate_finalize_ungrouped is the normalizing terminal variant.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, std::uint32_t store_lo, std::uint32_t store_hi, std::uint32_t idx_offset = 0>
inline void _gmg_merge16_to_run()
{
    _gmg_merge16_core<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    // Add the block's expert-id base offset (b*256) to the run's indices so they become GLOBAL ids.
    // idx lives in the LO16 of the LREG4/5 concat (score in HI16); idx+offset <= 511 never carries into
    // bit 16, so an int add to the whole concat shifts only the index, leaving the score untouched.
    if constexpr (idx_offset != 0)
    {
        // Raw integer add of the 12-bit immediate to LREG4/5 (the idx|score concat). Must use TTI (not
        // sfpi l_reg[]): sfpi's SSA register model doesn't write back to the physical LREG that the
        // surrounding raw TTI_SFPSTOREs read, so an sfpi add here is a no-op. SFPIADD with ARG_IMM does
        // lreg_dest = lreg_c + imm; idx (LO16) + offset <= 511 never carries into HI16 (score).
        TTI_SFPIADD(idx_offset, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_IMM);
        TTI_SFPIADD(idx_offset, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_IMM);
        TTI_SFPNOP;
    }
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + store_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + store_hi);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + store_lo);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + store_hi);
}

/**
 * @brief Terminal ungrouped finalize: merge the two sorted-8 runs at DEST columns {0,2}+{4,6} into the
 *        global top-8, keep the top-`topk`, normalize (softmax-over-selected or linear) and scale, then
 *        write the normalized scores and their expert indices back to the score/index regions.
 *
 * For 256 experts the two runs are topA/topB; for >256 they are the last two block/subtree runs of the
 * combine tree. SFPU (MATH). The softmax max-subtraction (bf16-overflow-safe) and the top-n masking
 * mechanics are documented inline in the body.
 *
 * @tparam APPROXIMATION_MODE: fast-approx mode forwarded to the bitonic sort and exp.
 * @tparam is_fp32_dest_acc_en: DEST accumulation mode; must be false.
 * @tparam topk: number of experts to keep (<= 8); ranks >= topk are zeroed before normalize. 8 = full top-8.
 * @tparam output_softmax: true = softmax over the selected top-k (exp before the mask, max-subtracted);
 *         false = linear renormalize (score / Σ).
 * @param eps: denominator stabilization added to the normalization sum.
 * @param scale: routed scaling factor applied to the normalized scores.
 * @note Calls @ref _gmg_merge16_core (which resets the Dst RWC counter and enables index tracking) and reads
 *       the two input runs at columns {0,2} and {4,6}; resets Dst RWC again internally before the softmax /
 *       normalize tails. Clobbers LREG0-7 (and LREG14). TERMINAL variant — it normalizes and writes the
 *       output; @ref _gmg_merge16_to_run is the non-normalizing run-producing variant.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, std::uint32_t topk = 8, bool output_softmax = false>
inline void _generalized_moe_gate_finalize_ungrouped(std::uint32_t eps, std::uint32_t scale)
{
    // topk is restricted to {4, 6, 8}: the rank-mask below is correct only for these. topk 1-3 fall into the
    // `topk <= 4` branch, which drops the offset-4 half but does NOT mask within offset-0, so they would
    // silently keep 4 ranks; topk 5/7 take the masked branch but are untested. The metal path already guards
    // this (host TT_FATAL + the unified kernel static_assert), but a direct LLK caller has no other gate --
    // fail loudly here too rather than emit silently-wrong output.
    static_assert(topk == 4 || topk == 6 || topk == 8, "topk must be one of {4, 6, 8} (rank-mask is correct only for these)");

    // Final combine: merge the two runs at {0,2}+{4,6} -> global top-8 (sorted DESCENDING), then normalize.
    // (For 256 the two runs are topA/topB; for >256 they are the last two block/subtree runs of the tree.)
    _gmg_merge16_core<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    bitonic_topk_store8_even_cols_split_indices_single_face<is_fp32_dest_acc_en>();

    // SOFTMAX OUTPUT (over the selected top-k): exp the sorted scores in place, BEFORE the top-n mask and
    // the normalize. linear-normalize(exp(s-M)) = exp(s_i)/Σexp(s) = softmax (the exp(M) factor cancels), so
    // the existing sum+recip+mul tail then yields softmax(selected) * scale. Must precede the mask so the
    // dropped ranks (zeroed by the mask afterwards) contribute 0 to Σexp, not exp(0)=1.
    //
    // MAX-SUBTRACTION (required, not optional): score_func="softmax" feeds the op RAW router logits
    // (enable_sigmoid=False, see tt_moe_gate.py), which are unbounded -- NOT in [0,1]. exp() of large logits
    // would overflow / saturate in bf16 (the sum can hit inf -> recip 0 -> NaN weights). Softmax is
    // shift-invariant, so subtract the selected max M before exp to pin every input <= 0 -> exp in (0,1] ->
    // no overflow, and Σexp in [1,k]. The selected max is rank 0: the top-8 are sorted DESCENDING, so the
    // global max sits at scores+0, SFPU lane 0. Broadcast it across the live ranks' lanes {0,8,16,24} with
    // the SFPCONFIG->LREG14 "replicate down the column" idiom (LREG14[lane] = LREG0[lane&7], so lane 0's M
    // lands on every lane with lane&7==0 -- exactly the 4 live ranks per row), then subtract via SFPMAD
    // (score + (-1)*M; LCONST_neg1 is the SFPU's -1.0 const, same use as ckernel_sfpu_rounding_ops.h). The
    // non-rank ("column != 0") lanes get junk-M and exp to garbage, but they are never summed (the tail's
    // reduction reads column 0 only) nor stored, so they are harmless. dst_reg[k] = TTI addr 2k (scores+0 ->
    // dst_reg[0], scores+4 -> dst_reg[2]); reset Dst RWC so the base lines up with the normalize's TTI loads.
    if constexpr (output_softmax)
    {
        // ---- subtract the global max (rank 0 @ scores+0 lane 0) from both score rows ----
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);                     // ranks 0-3; lane 0 = max M
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);                     // ranks 4-7
        TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);                                              // LREG14[lane] = LREG0[lane&7] -> M @ lanes 0/8/16/24
        TTI_SFPMAD(p_sfpu::LREG14, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0); // LREG0 = scores - M
        TTI_SFPMAD(p_sfpu::LREG14, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0); // LREG1 = scores - M
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);

        // ---- exp the max-shifted scores in place (inputs <= 0 -> exp in (0,1], no overflow) ----
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        sfpi::vFloat e0                        = sfpi::dst_reg[(scores_offset + 0) / 2];
        sfpi::dst_reg[(scores_offset + 0) / 2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(e0);
        sfpi::vFloat e4                        = sfpi::dst_reg[(scores_offset + 4) / 2];
        sfpi::dst_reg[(scores_offset + 4) / 2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(e4);
    }

    // TOP-N (n = topk <= 8): the sorted-8 are at scores/indices {0,4} — offset 0 = ranks 0-3, offset 4 =
    // ranks 4-7 (descending). Zero the scores+idx of ranks >= topk BEFORE the normalize so the denominator
    // is the sum of the top-n and the dropped slots output 0. topk==8 -> no-op (full top-8).
    if constexpr (topk <= 4)
    {
        // Drop the entire offset-4 half (ranks 4-7). (topk<4 would also need offset-0 lane masking.)
        TTI_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, scores_offset + 4);
        TTI_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, indices_offset + 4);
    }
    else if constexpr (topk < 8)
    {
        // TOP-5/6/7: the 4 sorted ranks 4-7 are spread every 8 lanes across the 32-lane offset-4 row
        // (lanes 0,8,16,24), and vConstTileId = 2*lane, so rank (4+j) -> tileid 16*j: rank4@0, rank5@16,
        // rank6@32, rank7@48 (pinned by dump: drop_thr 4/8/16 all keep only rank4, so the stride is 16).
        // Drop ranks topk..7 = lanes with tileid >= drop_thr (drop_thr = 16*(topk-4)); keep ranks
        // 4..topk-1. Use sfpi v_if for the per-lane predicate: it materializes vConstTileId
        // correctly (raw SFPIADD/SFPMOV reads of the LTILEID const-reg did NOT yield per-lane values and
        // zeroed the whole row). KEY: sfpi dst_reg[k] addresses TTI addr k*SFP_DESTREG_STRIDE (=k*2), and
        // its default mod-0 (SrcB) load/store matches the normalize's mod-0 TTI_SFPLOAD of the scores --
        // so scores+4 (TTI addr 4) is dst_reg[2], not dst_reg[4] (that earlier off-by-stride wrote to
        // addr 8 and left scores+4 untouched -> full top-8). Reset Dst RWC first so the dst_reg base
        // lines up with the TTI ADDR_MOD_3 base. Zero the dropped slots' idx too (matches the topk<=4
        // path), same read-modify-write the scores use: load the row, zero the dropped lanes in-register,
        // store all lanes back (kept lanes round-trip exactly). NOTE: the idx row must be read/written as
        // vFloat (the default mod-0 SrcB raw-bit passthrough), NOT vUInt -- a vUInt dst_reg store here
        // simply did not land. The mod-0 passthrough preserves the kept ids (e.g. 337/75) bit-for-bit.
        constexpr int drop_thr = 16 * (static_cast<int>(topk) - 4); // tileid >= drop_thr -> rank >= topk
        constexpr int sc4_dreg = (scores_offset + 4) / 2;           // scores+4 (TTI addr 4)  -> dst_reg 2
        constexpr int ix4_dreg = (indices_offset + 4) / 2;          // indices+4 (TTI addr 68) -> dst_reg 34
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        sfpi::vFloat sc = sfpi::dst_reg[sc4_dreg];
        v_if (sfpi::vConstTileId >= drop_thr)
        {
            sc = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[sc4_dreg] = sc;
        sfpi::vFloat ix         = sfpi::dst_reg[ix4_dreg];
        v_if (sfpi::vConstTileId >= drop_thr)
        {
            ix = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[ix4_dreg] = ix;
    }

    // ---- normalization tail (same as _generalized_moe_gate_top8) ----
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 0);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, interm_offset + 0);
    sfpi::vFloat l0                 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value          = Converter::as_float(eps);
    l0                              = l0 + eps_value;
    l0                              = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value        = Converter::as_float(scale);
    l0                              = l0 * scale_value;
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    TTI_SFPNOP;
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, scores_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_generalized_moe_gate_topk()
{
    // Intentional no-op: reciprocal is initialized inside top8 / finalize_ungrouped (not here). The gate
    // uses reg 14 to broadcast, which would clobber the reciprocal constants if they were initialized ahead
    // of time, so both WH and BH set them up within those fns. Kept (empty) because the topk_init wrapper
    // chain calls it on both arches.
}

} // namespace sfpu
} // namespace ckernel
