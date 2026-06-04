// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "lltt.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// TODO: Initial evaluation of using replay buffers here did not show any performance improvement
// Try re-evaluating with latest op sequence and record larger sequences

constexpr uint32_t dst_tile_offset = 64;  // 1 tile x 64 rows per tile
constexpr uint32_t scores_offset = 0;
constexpr uint32_t indices_offset = scores_offset + dst_tile_offset;
constexpr uint32_t bias_offset = indices_offset + dst_tile_offset;
constexpr uint32_t interm_offset = bias_offset + dst_tile_offset;

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16_single_face() {
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 12);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_offset + 12);
}

template <bool is_fp32_dest_acc_en, uint32_t offset = 0>
inline void bitonic_topk_load16_concat_indices_single_face() {
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
inline void bitonic_topk_load16_concatted_indices_single_face() {
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 12);

    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be false");
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store16_concatted_indices_single_face() {
    // Store 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 12);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store16_single_face() {
    // Store 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 12);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_even_cols_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_even_cols_concatted_indices_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_split_indices_single_face() {
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
inline void bitonic_topk_store8_even_cols_concatted_indices_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 4);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_odd_cols_concatted_indices_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 6);

    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 6);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_odd_cols_concatted_indices_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 2);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 6);

    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 2);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, interm_offset + 6);
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph0_st1_to_1_single_face() {
    if constexpr (start_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph1_st2_to_1_single_face() {
    if constexpr (start_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool end_transpose, bool bitonic = true>
inline void bitonic_topk_ph2_st3_to_1_single_face() {
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    if constexpr (bitonic) {
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    } else {
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

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool dir, bool end_transpose>
inline void bitonic_top8_ph3_st4_to_1() {
    // TODO: Use replay buffer for these instructions
    if constexpr (dir == (bool)SortDir::ArgMax) {
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
        if constexpr (end_transpose) {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    } else {
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

        if constexpr (end_transpose) {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir>
inline void bitonic_top8_ph0_to_ph3() {
    // Phase 0
    {
        constexpr bool start_transpose = true;
        constexpr bool end_transpose = false;
        constexpr int phase_replay_count = 2 + (int)start_transpose + (int)end_transpose;
        bitonic_topk_ph0_st1_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 1
    {
        constexpr bool start_transpose = false;
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 4 + (int)start_transpose + (int)end_transpose;
        // Odd Columns
        bitonic_topk_ph1_st2_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 2
    {
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 7 + (int)end_transpose;
        // Even Columns
        bitonic_topk_ph2_st3_to_1_single_face<end_transpose>();
    }
    // Modified Phase 3 for top8
    {
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 8 + (int)end_transpose;
        bitonic_top8_ph3_st4_to_1<idir, end_transpose>();
    }
}

void reverse_sort_order() {
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_sum_top2() {
    constexpr bool idir = false;  // Sort descending order
    constexpr int load_store_replay_count = 8;
    constexpr int load_replay_offset = 0;
    constexpr int store_replay_offset = load_replay_offset + load_store_replay_count;
    constexpr int phase_replay_offset = store_replay_offset + load_store_replay_count;

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
inline void _generalized_moe_gate_sort_top4_groups() {
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
template <bool is_fp32_dest_acc_en, uint32_t base = 0>
inline void _gmg_merge4_runs_raw() {
    constexpr bool idir = false;  // Sort descending order

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
inline void _generalized_moe_gate_top8(uint32_t eps, uint32_t scale) {
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
    sfpi::vFloat l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value = Converter::as_float(eps);
    l0 = l0 + eps_value;
    l0 = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    l0 = l0 * scale_value;
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

// ============================================================================
// UNGROUPED top-8 (DRAFT — needs HW validation against the golden test)
//
// Merges the 4 sorted-descending top-8 runs at columns {read_base, +2, +4, +6}
// into a single top-8 and stores it (bias value + LO16 index + HI16 score) at
// columns {store_lo, store_hi}. This is exactly the existing top8's 4-column
// merge (stage 1 + stage 2 of _generalized_moe_gate_top8), with the initial load
// offsets and the final store offsets parameterized. Stage 2 is base-independent.
//
// VERIFY on HW:
//  - that reading read_base=8 (groups 4-7) behaves identically to read_base=0
//  - that the interm-region scratch (cols 0,4) used by the two store/load helpers
//    does not collide between the two merge4 calls (it shouldn't: they run serially)
//  - that no transpose (step0/step1) is needed before consuming the sum_top2 layout
// ============================================================================
template <bool is_fp32_dest_acc_en, uint32_t read_base, uint32_t store_lo, uint32_t store_hi>
inline void _gmg_merge4_top8() {
    constexpr bool idir = false;  // descending
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

// Ungrouped global top-8 over all 8 groups (256 experts): true top-8 by bias score.
// Replaces step0 -> sort_top4 -> step1 -> top8 with: merge groups 0-3 and 4-7 into
// cols {0,4} and {2,6}, then reuse the existing top8 (merge cols 0,2,4,6 + normalize).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_top8_ungrouped(uint32_t eps, uint32_t scale) {
    _gmg_merge4_top8<is_fp32_dest_acc_en, 0, 0, 4>();  // groups 0-3 -> cols {0,4}
    _gmg_merge4_top8<is_fp32_dest_acc_en, 8, 2, 6>();  // groups 4-7 -> cols {2,6}
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

// Copy a stored top-8 run (bias + concat idx|score) from offset pair {from_lo,from_hi} to
// {to_lo,to_hi}, matching _gmg_merge4_top8's store convention (bias mode 0; idx LO16, score HI16).
// Used to relocate a saved run (e.g. topA parked at safe rows 8-15) into the final merge slot.
template <uint32_t from_lo, uint32_t from_hi, uint32_t to_lo, uint32_t to_hi>
inline void _gmg_copy_topk_run() {
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

// DIAGNOSTIC: normalize a top-8 run already sitting at scores/indices {0,4} (the normalize tail of
// _generalized_moe_gate_top8, factored out). Used by GMG_DIAG_TOPA to output a single half's run.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _gmg_normalize_run(uint32_t eps, uint32_t scale) {
    // Reset Dst RWC (see _gmg_copy_topk_run): a preceding FPU MOP leaves it advanced, biasing SFPLOAD.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
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
    sfpi::vFloat l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value = Converter::as_float(eps);
    l0 = l0 + eps_value;
    l0 = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    l0 = l0 * scale_value;
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

// Ungrouped finalize: merge TWO complete sorted-8 runs — topA at {0,2} (rows 0-3) and topB at
// {4,6} (rows 4-7) — into the global top-8, then normalize. Unlike _gmg_merge4_runs_raw (which
// treats {0,2,4,6} as 4 separate runs and would split topA/topB across the bitonic halves, causing
// duplicates), this loads each run WHOLE: topA -> LREG0/1+LREG4/5, topB -> LREG2/3+LREG6/7 (reversed),
// one bitonic_top8_ph3 stage merges the two sorted-8 runs, then the standard normalization tail runs.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _generalized_moe_gate_finalize_ungrouped(uint32_t eps, uint32_t scale) {
    constexpr bool idir = false;  // descending
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // reverse run = topB at {4,6}: lo (offset 4) -> LREG2, hi (offset 6) -> LREG3, then reverse.
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 6);
    reverse_sort_order();

    // forward run = topA at {0,2}.
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_3, scores_offset + 2);
    bitonic_top8_ph3_st4_to_1<idir, true>();
    bitonic_topk_store8_even_cols_split_indices_single_face<is_fp32_dest_acc_en>();

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
    sfpi::vFloat l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value = Converter::as_float(eps);
    l0 = l0 + eps_value;
    l0 = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    l0 = l0 * scale_value;
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

// ============================================================================
// P1 STORE-FOOTPRINT PROBE (unambiguous) — run AFTER sum_top2.
//
// Fills LREG0..3 with distinct constant markers (100/200/300/400) and stores them to the
// INDICES region at offsets {0,2,4,6} (the offsets the merge uses). Dumping the indices
// region then shows, with zero ambiguity, the *store footprint* of each offset: every face
// cell that became 100 belongs to offset 0's footprint, 200 -> offset 2, etc.; cells still
// holding an original expert id (0..255) were not covered by any of these four stores.
// This maps "SFPSTORE offset N -> which 32 face (row,col) cells" directly, independent of
// load geometry. (SFPLOAD footprint is the same set of cells by symmetry.)
// ============================================================================
inline void _gmg_probe_lanemap() {
    // SFPTRANSP probe: set LREG0..3 to uniform constants 1/2/3/4 (every lane of LREGi = i+1),
    // run one SFPTRANSP, then store the 4 LREGs to the rows-0-7 footprints (offsets 0/2/4/6).
    // Reveals SFPTRANSP's action: if it does (LREGi,row j)<->(LREGj,row i) and leaves the 8
    // SFPU columns (groups) untouched, the bias dump should show each face row uniform across
    // columns with row0=1,row1=2,row2=3,row3=4,row4=1,... (LREG0,1 -> rows0-3 even/odd;
    // LREG2,3 -> rows4-7). A per-column-uniform result means SFPTRANSP does NOT cross groups.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vFloat(1.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vFloat(2.0f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vFloat(3.0f);
    sfpi::l_reg[sfpi::LRegs::LReg3] = sfpi::vFloat(4.0f);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, bias_offset + 0);  // rows 0-3, even cols
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, bias_offset + 2);  // rows 0-3, odd cols
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, bias_offset + 4);  // rows 4-7, even cols
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, bias_offset + 6);  // rows 4-7, odd cols
}

// Move groups 4-7 over groups 0-3 (post-step0 layout: group c at face row c, even cols).
// Copies rows 4-7 (offset4) -> rows 0-3 (offset0), bit-exact (INT32), for bias/indices/scores.
// Run AFTER step0 (and skip sort_top4): the skip-sort path's step1 grabs rows 0-3, so after this
// the merge sees groups 4-7 instead of groups 0-3.
inline void _gmg_shift_hi_groups() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_3, scores_offset + 4);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_3, scores_offset + 0);
}

// COORDINATE PROBE: write a distinct constant (1000 + offset) to the indices region at each
// SFPLOAD/SFPSTORE offset, covering face0 (0,2,4,6 = rows0-3even/odd, rows4-7even/odd) and
// face1 (16,18,20,22). The dump then shows, for every display cell, which offset's store
// covered it (value-1000 = the offset). Cross-referenced with the known post-step0 group map,
// this resolves exactly which offset addresses groups 4-7. Run standalone after step0.
inline void _gmg_probe_offsets() {
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1000);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 0);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1002);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 2);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1004);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 4);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1006);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 6);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1016);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 16);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1018);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 18);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1020);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 20);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vInt(1022);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, indices_offset + 22);
}

// Rotate the 8 SFPU columns of the LREG at DEST `addr` LEFT by 4 (== swap halves {0-3}<->{4-7}),
// bit-exact (INT32), via 4 chained SHFLROR1 (each rotates the 8 columns by 1; 4x = by 4).
// LREG0 = data, LREG1 = scratch (avoid LREG4-7 / index-tracking HW bug).
template <uint32_t addr>
inline void _gmg_rot8by4_at() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, addr);
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, addr);
}

// Rotate the 8 groups by 4 (groups 4-7 -> front), at the post-sum_top2 layout (group g in
// face-0 even col 2g, top-8 over rows 0-7). Rotates even-col blocks of bias/indices/scores at
// offsets 0 (rows0-3) and 4 (rows4-7). Run AFTER sum_top2, BEFORE step0.
inline void _gmg_rotate_groups_by4() {
    // Rotate BOTH even (offset 0/4) and odd (offset 2/6) column blocks of rows 0-7, so whole
    // group column-pairs move together (rotating only even cols splits each group and scrambles).
    _gmg_rot8by4_at<scores_offset + 0>();
    _gmg_rot8by4_at<scores_offset + 2>();
    _gmg_rot8by4_at<scores_offset + 4>();
    _gmg_rot8by4_at<scores_offset + 6>();
    _gmg_rot8by4_at<indices_offset + 0>();
    _gmg_rot8by4_at<indices_offset + 2>();
    _gmg_rot8by4_at<indices_offset + 4>();
    _gmg_rot8by4_at<indices_offset + 6>();
    _gmg_rot8by4_at<bias_offset + 0>();
    _gmg_rot8by4_at<bias_offset + 2>();
    _gmg_rot8by4_at<bias_offset + 4>();
    _gmg_rot8by4_at<bias_offset + 6>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_generalized_moe_gate_topk() {
    // Note: For BH there is no conflict with reg usage between the gate and reciprocal
    // For WH, since we use reg 14 to broadcast, this would overwrite the recip value, so we init within the top8 fn
    // instead of ahead of time
    // sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
