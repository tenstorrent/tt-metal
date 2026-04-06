// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Selects the top-8 values (with expert indices) from 32 inputs within a single core.
 *
 * Operates on one DST tile containing 32 FP16B gate values arranged in 4 groups of
 * 8 rows. Each value represents a different expert's score for the tokens in that lane.
 *
 * Phase 1 - Sort: Each of the 4 groups (8 values) is sorted in descending order using
 *   a full **bitonic sort** network (24 SFPSWAPs: 4 stage-1 + 8 stage-2 + 12 stage-3).
 *   A 3-bit intra-group index (0-7) is fused into the lower 16 bits of each value before
 *   sorting, so indices travel with their values through swaps.
 * Phase 2 - Index encoding: A 2-bit group offset (0-3) is added to bits [4:3] of each
 *   index, extending the expert index to 5 bits (0-31).
 * Phase 3 - Merge: The 4 sorted-8 sequences are merged pairwise into a single top-8
 *   using an optimized bitonic merge (max(A[i], B[7-i]) followed by 12-swap merge),
 *   requiring only 3 sequential merge passes.
 * Phase 4 - Masking: A per-lane mask (from tile 2) is checked; if the bit for
 *   `tile_index` is unset, that lane's values are replaced with -infinity. A 3-bit
 *   core ID (`tile_index`) is encoded into bits [7:5] of each index.
 *
 * Input:  32 FP16B values per lane from the DST tile at `dst_index` (all 4 faces).
 *         A mask tile at DST offset 128 (tile 2).
 * Output: 8 FP16B values (top-8 descending) in face 0 rows 0-7, and 8 corresponding
 *         8-bit expert indices in face 0 rows 8-15 (LO16 mode), of the same DST tile.
 *         Index encoding: bits [7:5]=core_id, bits [4:3]=group, bits [2:0]=intra-group.
 */

namespace ckernel {

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "lltt.h"
#include "sfpi.h"

namespace sfpu {

//-----------------------------------------------------------------------------
// Bitonic Sort for 8 Values
//-----------------------------------------------------------------------------
//
// Bitonic sort for LREG0-7, result is DESCENDING (LREG0 = max, LREG7 = min)
// SFPSWAP(0, A, B, ALL_ROWS_MAX): Puts max(A,B) in A, min(A,B) in B
//
// Stage 1: Build alternating direction pairs (4 swaps)
// Stage 2: Merge pairs into sorted-4 sequences (10 swaps: 4 left + 4 right + 2 to complete right)
// Stage 3: Final bitonic merge of 8 to sorted descending (12 swaps)
//
// Total: 24 swaps (fits in 32-slot replay buffer)
//-----------------------------------------------------------------------------

inline void _top8_configure_addrmod_() {
    // SFPU configuration shifts to use ADDRMOD4-7.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = -22, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);

    addr_mod_t{
        .dest = {.incr = 2, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_1);

    addr_mod_t{
        .dest = {.incr = 14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = -14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

// Load input data (FP16B) from tile layout and fuse with expert indices
inline void _load_input_group_() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);  // offset 22
}

inline void _add_row_indices_() {
    // Fuse expert indices into LO16 bits
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 1);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 2);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 3);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 4);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 5);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 6);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 7);
}

// Store sorted group back to tile layout
inline void _store_sorted_group_() {
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);  // offset 22

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, 64);  // offset 22
}

// Load sorted group from tile layout
inline void _load_sorted_group_() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);  // offset 22

    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, 64);  // offset 22
}

template <uint32_t face_offset, uint32_t group_offset, uint32_t row_offset>
inline void _top8_set_d_rwc_() {
    constexpr uint32_t total_offset = face_offset * 32 + group_offset + row_offset;

    if constexpr (total_offset < 16) {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, total_offset, 0, 0, p_setrwc::SET_D);
        return;
    }

    constexpr uint32_t num_incr = (total_offset / 8) - 1;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 8 + row_offset, 0, 0, p_setrwc::SET_D);

    for (uint32_t i = 0; i < num_incr; ++i) {
        TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    }
}

// Load sorted group from tile layout
template <uint32_t face_offset, uint32_t group_offset>
inline void _top8_load_rows_0_3_() {
    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 0>();

    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18

    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 0>();

    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
}

template <uint32_t face_offset, uint32_t group_offset>
inline void _top8_load_rows_4_7_() {
    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 4>();

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 22

    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 4>();

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 22
}

template <uint32_t face_offset, uint32_t group_offset>
inline void _top8_store_rows_0_3_() {
    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 0>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18

    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 0>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
}

template <uint32_t face_offset, uint32_t group_offset>
inline void _top8_store_rows_4_7_() {
    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 4>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 22

    _top8_set_d_rwc_<face_offset, group_offset, /*row_offset*/ 4>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 22
}

inline void _top8_load_indices_() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, 64);  // offset 22
}

inline void _top8_store_indices_() {
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 64);  // offset 18
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 64);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 64);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, 64);  // offset 22
}

template <uint32_t group_offset>
inline void _top8_add_group_offset_to_indices_() {
    // Add the group offset to LREG0 to LREG7, in bits 3 and 4 from the lower 16 bits
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        group_offset << 3, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
}

inline void _top8_add_core_offset_to_indices_0_3_(uint32_t core_offset) {
    // Add the core offset to LREG0 to LREG3, in bits 7-5 from the lower 16 bits
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
}

inline void _top8_add_core_offset_to_indices_4_7_(uint32_t core_offset) {
    // Add the core offset to LREG4 to LREG7, in bits 7-5 from the lower 16 bits
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TT_SFPIADD(core_offset << 5, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
}

inline void _top8_store_rows_0_3_final_() {
    _top8_set_d_rwc_<0, 0, /*row_offset*/ 0>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18

    _top8_set_d_rwc_<0, 8, /*row_offset*/ 0>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 0);  // offset 18
}

inline void _top8_store_rows_4_7_final_() {
    _top8_set_d_rwc_<0, 0, /*row_offset*/ 4>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 22

    _top8_set_d_rwc_<0, 8, /*row_offset*/ 4>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 0);  // offset 22
}

//-----------------------------------------------------------------------------
// Bitonic Merge for Already-Bitonic Sequence (12 swaps)
//-----------------------------------------------------------------------------
//
// Given 8 values in LREG0-7 that form a bitonic sequence (e.g., valley-shaped
// from the merge operation max(A[i], B[7-i])), sort them into descending order.
// This requires only 12 swaps vs 24 for full bitonic sort.
//
//-----------------------------------------------------------------------------
inline void _top8_bitonic_merge_8_rows_() {
    // Stage 1: Compare distance 4 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Stage 2: Compare distance 2 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Stage 3: Compare distance 1 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
}

inline void _top8_bitonic_sort_8_rows_() {
    // Stage 1: Build alternating pairs (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);  // 0>1 desc
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);  // 2<3 asc
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);  // 4>5 desc
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);  // 6<7 asc

    // Stage 2: Merge into sorted-4 sequences (8 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    // Stage 3: Final bitonic merge (12 swaps)
    _top8_bitonic_merge_8_rows_();
}

//-----------------------------------------------------------------------------
// BITONIC MERGE: Two sorted-8 → Top-8
//-----------------------------------------------------------------------------
//
// Given A[0-7] sorted desc in LREG, B[0-7] sorted desc at b_row (tile layout):
//   top-8 candidates = max(A[i], B[7-i]) for i=0..7
//   Then bitonic sort those 8 for final order.
//
// Tile layout index mapping for B[i]:
//   B[0]=+0, B[1]=+2, B[2]=+16, B[3]=+18, B[4]=+4, B[5]=+6, B[6]=+20, B[7]=+22
//
// So B[7-i] reversed mapping:
//   i=0: B[7]=+22, i=1: B[6]=+20, i=2: B[5]=+6, i=3: B[4]=+4
//   i=4: B[3]=+18, i=5: B[2]=+16, i=6: B[1]=+2, i=7: B[0]=+0
//
//-----------------------------------------------------------------------------

template <uint32_t face_offset, uint32_t grp_offset, bool replay>
inline void _top8_merge_sorted_8_() {
    // First 4 comparisons: A[0-3] vs B[7,6,5,4]
    // A[0-3] at tile offsets: 0, 2, 16, 18
    // B[7,6,5,4] at tile offsets: 22, 20, 6, 4

    // Load A[0-3] to LREG0-3
    _top8_load_rows_0_3_</*face_offset*/ 0, /*group_offset*/ 0>();

    // Load B[4-7] to LREG4-7 (LREG4=B[4], LREG5=B[5], LREG6=B[6], LREG7=B[7])
    _top8_load_rows_4_7_<face_offset, grp_offset>();

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Compare reversed: A[0] vs B[7], A[1] vs B[6], A[2] vs B[5], A[3] vs B[4]
    // max goes to LREG0-3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store first 4 winners
    _top8_store_rows_0_3_</*face_offset*/ 0, /*group_offset*/ 0>();

    // Second 4 comparisons: A[4-7] vs B[3,2,1,0]
    // A[4-7] at tile offsets: 4, 6, 20, 22
    // B[3,2,1,0] at tile offsets: 18, 16, 2, 0

    // Load A[4-7] to LREG4-7 (LREG4=A[4], LREG5=A[5], LREG6=A[6], LREG7=A[7])
    _top8_load_rows_4_7_</*face_offset*/ 0, /*group_offset*/ 0>();

    // Load B[0-3] to LREG0-3 (LREG0=B[0], LREG1=B[1], LREG2=B[2], LREG3=B[3])
    _top8_load_rows_0_3_<face_offset, grp_offset>();

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Compare reversed: A[4] vs B[3], A[5] vs B[2], A[6] vs B[1], A[7] vs B[0]
    // max goes to LREG4-7
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store second 4 winners
    _top8_store_rows_4_7_</*face_offset*/ 0, /*group_offset*/ 0>();

    // Load all 8 winners and bitonic merge (only 12 swaps needed for bitonic sequence)
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    if constexpr (replay) {
        lltt::replay(0, 16);
        TTI_SFPTRANSP(0, 0, 0, 0);
        _top8_bitonic_merge_8_rows_();
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(16, 16);
    } else {
        lltt::record<lltt::Exec>(0, 16);
        _load_sorted_group_();
        TTI_SFPTRANSP(0, 0, 0, 0);
        _top8_bitonic_merge_8_rows_();
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::record<lltt::Exec>(16, 16);
        _store_sorted_group_();
    }
}

inline void _calculate_top8_tile_(uint32_t tile_index) {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    //-------------------------------------------------------------------------
    // PHASE 1: Sort all 4 groups using replay buffer
    //-------------------------------------------------------------------------
    lltt::record<lltt::Exec>(0, 8);
    _load_input_group_();
    TTI_SFPTRANSP(0, 0, 0, 0);
    _add_row_indices_();
    lltt::record<lltt::Exec>(8, 24);
    _top8_bitonic_sort_8_rows_();
    TTI_SFPTRANSP(0, 0, 0, 0);
    _store_sorted_group_();

    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);

    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _add_row_indices_();
    lltt::replay(8, 24);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _store_sorted_group_();

    // Need to increment the dst index by 16 for group 2
    // Have to increment twice because the increment field is only 4 bit wide :(
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);

    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _add_row_indices_();
    lltt::replay(8, 24);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _store_sorted_group_();

    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);

    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _add_row_indices_();
    lltt::replay(8, 24);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _store_sorted_group_();

    //-------------------------------------------------------------------------
    // Phase 2: Add group offset to the indices
    //-------------------------------------------------------------------------
    // No need to touch group 0 has it already has 0 as the group offset

    // Group 1
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 8, 0, 0, p_setrwc::SET_D);
    lltt::record<lltt::Exec>(0, 9);
    _top8_load_indices_();
    TTI_SFPTRANSP(0, 0, 0, 0);
    _top8_add_group_offset_to_indices_<1>();
    lltt::record<lltt::Exec>(9, 9);
    TTI_SFPTRANSP(0, 0, 0, 0);
    _top8_store_indices_();

    // Move to lower half
    // Need to increment twice because the increment field is only 4 bit wide :(
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);

    // Group 2
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    lltt::replay(0, 9);
    _top8_add_group_offset_to_indices_<2>();
    lltt::replay(9, 9);

    // Group 3
    TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    lltt::replay(0, 9);
    _top8_add_group_offset_to_indices_<3>();
    lltt::replay(9, 9);

    //-------------------------------------------------------------------------
    // PHASE 3: Sequential merge all 4 groups into final top-8
    //-------------------------------------------------------------------------
    // Group 0 is already sorted in place from Phase 1, merge with groups 1-3
    _top8_merge_sorted_8_</*face_offset*/ 0, /*group_offset*/ 8, /*replay=*/false>();
    _top8_merge_sorted_8_</*face_offset*/ 1, /*group_offset*/ 0, /*replay=*/true>();
    _top8_merge_sorted_8_</*face_offset*/ 1, /*group_offset*/ 8, /*replay=*/true>();

    //-------------------------------------------------------------------------
    // PHASE 4: Flush some lanes if this group is not selected
    //-------------------------------------------------------------------------
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // For these 4 indices, put the core_id in the LO 16 bits (bits 7-5)
    _top8_add_core_offset_to_indices_0_3_(tile_index);

    // Mask is available in tile 2
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 128);

    // Now, let us bitshift by tile_index to get the mask for this tile
    // Mask to get only bit at tile_index
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 1 << tile_index);
    TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // LREG4 now has zero or (1<<tile_index)

    // Enable lanes if LREG4 is 0
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);

    // Load negative infinity constant (BF16 format)
    constexpr uint16_t NEG_INF_BF16 = 0xFF80;

    // Conditionally load -inf into LREG0-3 (only affects lanes where condition is true)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);

    // Clear condition codes (re-enable all lanes)
    TTI_SFPENCC(0, 0, 0, 0);

    // Store the result
    TTI_SFPTRANSP(0, 0, 0, 0);
    _top8_store_rows_0_3_final_();

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    _top8_load_rows_4_7_</*face_offset*/ 0, /*group_offset*/ 0>();

    TTI_SFPTRANSP(0, 0, 0, 0);

    _top8_add_core_offset_to_indices_4_7_(tile_index);

    // Set condition code to 0 if LREG0 is 0
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);

    // Conditionally load -inf into LREG4-7 (only affects lanes where condition is true)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);

    // Clear condition codes (re-enable all lanes)
    TTI_SFPENCC(0, 0, 0, 0);

    // Store the result
    TTI_SFPTRANSP(0, 0, 0, 0);
    _top8_store_rows_4_7_final_();
}

}  // namespace sfpu

inline void _llk_math_top8_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(ckernel::sfpu::_top8_configure_addrmod_);
}

inline void _llk_math_top8_tile_(uint32_t tile_index, uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_calculate_top8_tile_, dst_index, VectorMode::RC_custom, tile_index);
}

#endif

/**
 * @brief Initializes the per-core top-8 selection SFPU operation.
 */
inline void top8_tile_init() { MATH((_llk_math_top8_tile_init_())); }

ALWI void top8_tile(uint32_t tile_index, uint32_t dst_index) { MATH((_llk_math_top8_tile_(tile_index, dst_index))); }

}  // namespace ckernel
