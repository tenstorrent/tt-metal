// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compute_kernel_api/common_globals.h"

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
// Total: 26 swaps (fits in 32-slot replay buffer)
//-----------------------------------------------------------------------------

inline void _top8_bitonic_configure_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_0);
}

inline void _bitonic_sort_8_swaps_() {
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

    // Complete right-block merge: ensure 4<=7 and 5<=6 (otherwise right half isn't ascending)
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    // Stage 3: Final bitonic merge (12 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
}

inline uint32_t _get_group_value_row_(uint32_t tile, uint32_t half, uint32_t grp) {
    return tile * 64 + half * 32 + grp * 8;
}

inline uint32_t _get_group_index_row_(uint32_t tile, uint32_t half, uint32_t grp) {
    return 8 * 64 + tile * 64 + half * 32 + grp * 8;
}

// Load input data (FP16B) from tile layout and fuse with expert indices
inline void _load_input_group_(uint32_t tile, uint32_t half, uint32_t grp) {
    uint32_t row_base = _get_group_value_row_(tile, half, grp);

    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 2);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 16);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 18);
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 4);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 6);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 20);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, row_base + 22);
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Fuse expert indices into LO16 bits
    uint32_t idx = tile * 32 + half * 16 + grp * 8;
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, idx + 0);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, idx + 1);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, idx + 2);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, idx + 3);
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, idx + 4);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, idx + 5);
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, idx + 6);
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, idx + 7);
}

// Store sorted group back to tile layout
inline void _store_sorted_group_(uint32_t tile, uint32_t half, uint32_t grp) {
    uint32_t value_base = _get_group_value_row_(tile, half, grp);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 2);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 16);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 18);
    TT_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 4);
    TT_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 6);
    TT_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 20);
    TT_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 22);

    uint32_t index_base = _get_group_index_row_(tile, half, grp);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 2);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 16);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 18);
    TT_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 4);
    TT_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 6);
    TT_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 20);
    TT_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 22);
}

// Load 8 values from tile layout into LREG0-7
inline void _load_sorted_group_(uint32_t tile, uint32_t half, uint32_t grp) {
    uint32_t value_base = _get_group_value_row_(tile, half, grp);

    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 2);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 16);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 18);
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 4);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 6);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 20);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base + 22);

    uint32_t index_base = _get_group_index_row_(tile, half, grp);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 2);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 16);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 18);
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 4);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 6);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 20);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base + 22);
    TTI_SFPTRANSP(0, 0, 0, 0);
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
// Uses scratch_row (tile layout) for temporary storage.
//-----------------------------------------------------------------------------

inline void _merge_sorted_8_(uint32_t tile, uint32_t half, uint32_t grp) {
    // First 4 comparisons: A[0-3] vs B[7,6,5,4]
    // A[0-3] at tile offsets: 0, 2, 16, 18
    // B[7,6,5,4] at tile offsets: 22, 20, 6, 4

    // Load A[0-3] to LREG0-3
    uint32_t value_base_A = _get_group_value_row_(0, 0, 0);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 2);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 16);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 18);

    uint32_t index_base_A = _get_group_index_row_(0, 0, 0);
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 0);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 2);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 16);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 18);

    // Load B[7-4] to LREG4-7
    uint32_t value_base_B = _get_group_value_row_(tile, half, grp);
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 22);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 20);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 6);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 4);

    uint32_t index_base_B = _get_group_index_row_(tile, half, grp);
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 22);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 20);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 6);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 4);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Compare: max goes to LREG0-3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Store first 4 winners
    TTI_SFPTRANSP(0, 0, 0, 0);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 2);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 16);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 18);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 2);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 16);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 18);

    // Second 4 comparisons: A[4-7] vs B[3,2,1,0]
    // A[4-7] at tile offsets: 4, 6, 20, 22
    // B[3,2,1,0] at tile offsets: 18, 16, 2, 0

    // Load A[4-7] to LREG0-3
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 4);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 6);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 20);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 22);

    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 4);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 6);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 20);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 22);

    // Load B[3,2,1,0] to LREG4-7
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 18);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 16);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 2);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_B + 0);

    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 18);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 16);
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 2);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_B + 0);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Compare: max goes to LREG0-3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Store second 4 winners
    TTI_SFPTRANSP(0, 0, 0, 0);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 4);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 6);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 20);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, value_base_A + 22);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 4);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 6);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 20);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, index_base_A + 22);

    // Load all 8 winners and bitonic sort
    _load_sorted_group_(0, 0, 0);
    lltt::replay(0, 26);
    _store_sorted_group_(0, 0, 0);
}

inline void _calculate_top8_bitonic_() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    //-----------------------------------------------------------------------------
    // PHASE 1: Sort all 32 groups using replay buffer
    //-----------------------------------------------------------------------------
    lltt::record<lltt::NoExec>(0, 26);
    _bitonic_sort_8_swaps_();

    for (uint32_t tile = 0; tile < 8; tile++) {
        for (uint32_t half = 0; half < 2; half++) {
            for (uint32_t grp = 0; grp < 2; grp++) {
                _load_input_group_(tile, half, grp);
                lltt::replay(0, 26);
                _store_sorted_group_(tile, half, grp);
            }
        }
    }

    //-----------------------------------------------------------------------------
    // PHASE 2: Sequential merge all 32 groups into final top-8
    //-----------------------------------------------------------------------------
    // Merge with remaining 31 groups
    for (uint32_t tile = 0; tile < 8; tile++) {
        for (uint32_t half = 0; half < 2; half++) {
            for (uint32_t grp = 0; grp < 2; grp++) {
                if ((tile == 0) && (half == 0) && (grp == 0)) {
                    continue;  // Nothing to merge with yet
                }
                _merge_sorted_8_(tile, half, grp);
            }
        }
    }

    // Write the indices to the first 8 rows of the output tile
    uint32_t index_base = 0;
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 0);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 2);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 16);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 18);
    TT_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 4);
    TT_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 6);
    TT_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 20);
    TT_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, index_base + 22);
}

}  // namespace sfpu

inline void _llk_math_top8_bitonic_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_top8_bitonic_configure_addrmod_);
}

inline void _llk_math_top8_bitonic_tile_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_calculate_top8_bitonic_, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the bitonic top-8 calculation.
 */
inline void top8_bitonic_tile_init() { MATH((_llk_math_top8_bitonic_tile_init_())); }

/**
 * @brief Calculates top-8 indices using optimized bitonic sort network.
 *
 * Algorithm:
 * - Phase 1: Sort each of 32 groups (8 values each) using bitonic sort
 *   - 26 swap instructions recorded in replay buffer, replayed 32 times
 * - Phase 2: Merge all groups sequentially into final top-8
 *   - Uses max(A[i], B[7-i]) selection then re-sort
 *
 * @param dst_index The destination tile index
 */
ALWI void top8_bitonic_tile(uint32_t dst_index) { MATH((_llk_math_top8_bitonic_tile_(dst_index))); }

}  // namespace ckernel
