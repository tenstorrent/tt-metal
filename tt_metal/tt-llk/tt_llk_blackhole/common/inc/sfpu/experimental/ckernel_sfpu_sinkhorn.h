// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU fast-path for sinkhorn normalization on 4x4 sub-matrices.
//
// Processes (4,4) sub-mats densely packed in-DST, avoiding per-step
// pack/unpack round-trips and letting SFPU SIMD do the work.
//
// Comb path: sinkhorn_row_max_sub subtracts each 4x4 row's max in DEST
// before exp_tile (softmax is shift-invariant per row; this keeps exp
// finite on large comb logits).
//
// LANE LAYOUT (per SFPLOAD ISA):
//   Row = (Addr & ~3) + (Lane / 8)
//   Col = (Lane & 7) * 2 + ((Addr & 2) ? 1 : 0)
// One LREG = 4 rows x 8 cols (one parity).
//
// For a 4x4 sub-mat at the top-left of face 0 (rows 0..3, cols 0..3):
//   LREG0 (face 0 even cols, addr=0): lanes {0,8,16,24} = col 0; {1,9,17,25} = col 2
//   LREG1 (face 0 odd cols,  addr=2): lanes {0,8,16,24} = col 1; {1,9,17,25} = col 3
// Padding cols (4..15) sit at lanes 2..7 of each subgroup and equal 0.
//
// ROW-NORM via 2-lane pair sum:
//   Pair-sum = even+odd LREGs of a strip (lanewise). Each 4-col sub-mat then
//   occupies a 2-lane pair of that pair-sum; fold {2i, 2i+1} so both lanes
//   hold the sub-mat row sum. Reciprocal + multiply applies the row-norm.
//   The fold computes odd lanes with one ROR1+ADD, then even lanes with
//   ROL1 (7x ROR1 within the 8-lane group) so even lane 2i gets the
//   already-folded odd-lane sum. The two strips' SHFT2s are independent
//   and interleaved. One shared SETCC/conditional-MOV merge
//   (parity mask in LREG11) completes the broadcast.
//
// COL-NORM via SFPTRANSP:
//   SFPTRANSP transposes the leftmost two axes of a 4x4x8 tensor view of
//   LReg[0..3], swapping the LReg-index axis with the lane/8 (row) axis.
//   After TRANSP, new[i][j*8+c] = old[j][i*8+c]. With our loads:
//     new LREG0 = (strip0 row 0 even cols, strip0 row 0 odd cols,
//                  strip1 row 4 even cols, strip1 row 4 odd cols)
//     new LREG1 = (strip0 row 1 ..., strip1 row 5 ...)
//     new LREG2 = (strip0 row 2 ..., strip1 row 6 ...)
//     new LREG3 = (strip0 row 3 ..., strip1 row 7 ...)
//   Cross-LREG lanewise sum (LREG0+LREG1+LREG2+LREG3) gives col sums per
//   strip at each lane. Multiply by recip → col-normalized in-place.
//   TRANSP back to restore original layout.
//
// Row-norm is the 2-lane pair fold; col-norm uses SFPTRANSP, extended
// across all 4 row-strips per face × 4 faces for the full tile.
//
// PERSISTENT CONSTANTS (programmed once per tile, before the iter loop):
//   LREG11 = lane-parity mask: lane k -> (2k & 2), i.e. 2 at odd lanes, 0 at
//            even lanes. Built per-column: SFPCONFIG copies LREG0 lanes 0..7
//            into LREG11 with a vertical broadcast across the 4 row-groups
//            (the 0x5555 instance mask's even bits select all 8 columns).
//   LREG13 = eps (fp32), guarding 0/0 in the row/col reciprocals.
// Both live outside LREG0..7, so SFPTRANSP (which permutes the {0..3} and
// {4..7} groups) never disturbs them and no per-iteration reloads exist.
// exp_tile_init reprograms LREG12..14 every tile, so the constants are
// (re)programmed at the top of each _sinkhorn_4x4_ call, after exp ran.
//
// Blackhole-only: SHFT2→dependent-consumer pairs rely on the Blackhole
// hardware auto-stall (on Wormhole explicit SFPNOPs would be required).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{

namespace sfpu
{

template <std::uint32_t VALID_H, std::uint32_t VALID_W, std::uint32_t ADDR = 0>
inline void _sinkhorn_zero_rect_padding_addrs_()
{
    constexpr std::uint32_t face     = ADDR / 16;
    constexpr std::uint32_t face_row = (face / 2) * 16;
    constexpr std::uint32_t face_col = (face % 2) * 16;
    constexpr std::uint32_t row_base = face_row + ((ADDR % 16) & ~3u);
    constexpr std::uint32_t parity   = (ADDR & 2) ? 1 : 0;

    constexpr std::uint32_t valid_rows          = VALID_H <= row_base ? 0 : VALID_H >= row_base + 4 ? 4 : VALID_H - row_base;
    constexpr std::uint32_t valid_width_in_face = VALID_W <= face_col ? 0 : VALID_W >= face_col + 16 ? 16 : VALID_W - face_col;
    constexpr std::uint32_t valid_cols          = valid_width_in_face <= parity ? 0 : (valid_width_in_face - parity + 1) / 2;

    if constexpr (valid_rows == 0 || valid_cols == 0)
    {
        TTI_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, ADDR);
    }
    else if constexpr (valid_rows < 4 || valid_cols < 8)
    {
        sfpi::vFloat value = sfpi::dst_reg[ADDR / 2];
        if constexpr (valid_cols < 8)
        {
            sfpi::vUInt local_col = sfpi::vConstTileId & sfpi::vUInt(0xE);
            v_if (local_col >= 2 * valid_cols)
            {
                value = 0.0f;
            }
            v_endif;
        }
        if constexpr (valid_rows < 4)
        {
            v_if (sfpi::vConstTileId >= 16 * valid_rows)
            {
                value = 0.0f;
            }
            v_endif;
        }
        sfpi::dst_reg[ADDR / 2] = value;
    }

    if constexpr (ADDR < 62)
    {
        _sinkhorn_zero_rect_padding_addrs_<VALID_H, VALID_W, ADDR + 2>();
    }
}

// After elementwise exp, zero everything outside the compile-time logical
// rectangle before any row/column reduction.
template <std::uint32_t VALID_H, std::uint32_t VALID_W>
inline void _sinkhorn_zero_rect_padding_()
{
    static_assert(VALID_H >= 1 && VALID_H <= 32);
    static_assert(VALID_W >= 1 && VALID_W <= 32);
    if constexpr (VALID_H < 32 || VALID_W < 32)
    {
        _sinkhorn_zero_rect_padding_addrs_<VALID_H, VALID_W>();
    }
}

// Program LREG11 = lane-parity mask: lane k -> (2k & 2), i.e. 2 at odd
// lanes, 0 at even. Used by the row-norm 2-lane pair fold (persistent
// across iters). Comb row-max builds a throwaway copy in a scratch LREG
// instead, so LCONST_neg1 (LREG11) stays live for its subtract.
//
// SFPCONFIG with a lane-masked LREG target copies LReg[0][lane & 7] into
// LREG[dest][lane] (per-column values, vertically broadcast across the 4
// row-groups); imm16 bit (2c) gates column c, so 0x5555 selects all columns.
//
// LCONST_neg1 aliases LREG11, so this clobbers the HW -1.0 constant. The
// row-norm body must not use LCONST_neg1 while the mask is live.
inline void _sinkhorn_program_parity_mask_()
{
    // Parity mask: LREG0 lane k -> 2k & 2 = (2 at odd lanes, 0 at even).
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 2);
    // SFPAND with Mod1=0 does VD = VD & VC; VC=15 is the read-only lane-id
    // register (LReg[15][k] = 2k).
    TTI_SFPAND(0, /*VC=*/15, p_sfpu::LREG0, 0);
    TTI_SFPCONFIG(0x5555, /*LREG11=*/11, /*MOD1_IMM16_IS_LANE_MASK=*/8);
}

// Program the persistent constants (see file header): LREG11 = lane-parity
// mask, LREG13 = eps. Called once per tile (exp_tile_init reprograms
// LREG12..14 every tile, so this must run after exp, inside the tile loop).
// EPS_BITS is the fp32 bit pattern of the eps constant.
template <std::uint32_t EPS_BITS>
inline void _sinkhorn_program_constants_()
{
    _sinkhorn_program_parity_mask_();

    // eps fp32 bits -> LREG13 (two 16-bit halves, then the config copy).
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (EPS_BITS >> 16) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, EPS_BITS & 0xFFFF);
    TTI_SFPCONFIG(0x5555, /*LREG13=*/13, /*MOD1_IMM16_IS_LANE_MASK=*/8);
}

// Per-row max-sub for the comb 4x4 at face-0 top-left (rows 0..3, cols 0..3).
// Must run after copy_tile and before exp_tile so exp(logit - row_max) cannot
// overflow on large comb logits. Softmax is shift-invariant per row.
//
// Same 2-lane pair geometry as row-norm: each 4-col chunk occupies lanes
// {2i, 2i+1} of the even/odd pair-max. Pair 0 is the comb 4x4; pairs 1..3
// are padding and are left alone (zeroed after exp). Does not touch
// LREG12..14 (exp_tile_init constants) or LREG11 (LCONST_neg1).
inline void _sinkhorn_row_max_sub_4x4_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);

    // Lanewise max(even, odd): lane 0 of each row = max(c0,c1), lane 1 = max(c2,c3).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // LREG5 (pair-min) is dead. Build throwaway parity there so LREG11
    // stays LCONST_neg1; interleave with the odd-lane fold.
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 2);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPAND(0, /*VC=*/15, p_sfpu::LREG5, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // Even-lane fold: LREG6 = ROL1(LREG4). ROL1 within an 8-lane group ==
    // 7xROR1; puts the odd-lane row max at even lanes.
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    // Merge even lanes from the ROL result. SETCC EQ0 on scratch parity
    // flags even lanes; SHFT2/SWAP/MOV do not disturb LaneFlags.
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    // LREG0/1 -= row_max. LREG11 was never clobbered, so LCONST_neg1 is live.
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
}

// One iter of row-norm + col-norm on a strip-pair already loaded in LREG0..3
// (two adjacent 4-row strips of a face, one LREG per strip per col parity).
// LREG4..7 are scratch; LREG11 holds the lane-parity mask and LREG13 the eps
// constant (both programmed per tile and untouched by SFPTRANSP, so nothing
// is reloaded per iteration).
inline void _sinkhorn_strip_pair_body_()
{
    // ── Row-norm: pair_sum + per-pair sum + recip + multiply. ──────────
    //
    // For 2D dense packing (multiple sub-mats horizontally per face row),
    // the row sum must be PER (4-col chunk) — not across all 16 cols of
    // the face. Each sub-mat occupies a 4-col chunk = 2 lanes of pair_sum.
    //
    // The fold broadcasts per-pair sums to BOTH lanes of each pair, so that
    // LREG_X[2i] = LREG_X[2i+1] = pair_sum[2i] + pair_sum[2i+1] = sub-mat
    // i's row sum.
    //
    // For vertical-only layouts (logical_w=4, cols 4..15 zero-padded),
    // sub-mats B,C,D have row-sum = 0 → recip(0+eps) = 1/eps = huge, but
    // the data at those lanes is also 0, so 0 × huge = 0 (preserved).
    //
    // Strip 0 pair sums into LREG4; strip 1 into LREG5.
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG5, 0);

    // Odd-lane fold: S = PS + ROR1(PS) is the sub-mat row sum at odd lanes
    // (lane 2i+1 <- PS[2i+1] + PS[2i]); even lanes stay garbage for now.
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG5, 0);

    // Even-lane fold: LREG6 = ROL1(LREG4), LREG7 = ROL1(LREG5). ROL1 within
    // an 8-lane group == 7xROR1; correct (sum_odd[2i+1]) at even lanes 2i.
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    // Merge even lanes from the ROL results. SETCC EQ0 on the parity mask
    // flags even lanes; SHFT2/ADD/MOV do not disturb LaneFlags, so both
    // gated MOVs share one SETCC.
    TTI_SFPSETCC(0, p_sfpu::LREG11, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG5, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    // Add eps to avoid 0/0 when row sum = 0 (e.g., padding strips zeroed
    // in DEST after exp).
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG5, 0);

    // Reciprocal: LREG4 = 1 / strip0_row_sum, LREG5 = 1 / strip1_row_sum.
    TTI_SFPARECIP(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPARECIP(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPARECIP_MOD1_RECIP);

    // Apply row recip to each strip's data LREGs.
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

    // ── Col-norm via SFPTRANSP. ────────────────────────────────────────
    //
    // After SFPTRANSP: new[i][j*8+c] = old[j][i*8+c]. SFPTRANSP permutes
    // both LREG[0..3] and LREG[4..7]; LREG11/LREG13 are outside both groups
    // and survive untouched.
    //
    // LREG mapping post-TRANSP for our load:
    //   new LREG0 = (strip0 row 0 across 16 cols, strip1 row 4 across 16 cols)
    //   new LREG1 = (strip0 row 1 ..., strip1 row 5 ...)
    //   new LREG2 = (strip0 row 2 ..., strip1 row 6 ...)
    //   new LREG3 = (strip0 row 3 ..., strip1 row 7 ...)
    // Lanes 0..15 = strip 0; lanes 16..31 = strip 1.
    //
    // Col sum per (strip, col) = lanewise SFPADD across LREG0..3.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // col_sum into LREG4: LREG4 = LREG0 + LREG1 + LREG2 + LREG3 lanewise.
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG4, 0);

    // Add eps to col_sum, then reciprocal: LREG4 = 1 / (col_sum + eps).
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPARECIP(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPARECIP_MOD1_RECIP);

    // Apply col recip to each LREG (which now holds one row across cols).
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

    // TRANSP back to restore the original (row-strip, parity) layout.
    TTI_SFPTRANSP(0, 0, 0, 0);
}

// Multi-iter strip-pair: load LREG0..3 once, run ``iters`` row+col norm passes
// entirely in fp32 lregs, then RNE-round and store once at the end — one bf16
// conversion per strip-pair, so no compounded per-iter truncation loss.
//
// addrs are face-relative offsets:
//   strip 0 even cols at addr_e0, strip 0 odd cols at addr_o0 (= addr_e0 + 2),
//   strip 1 even cols at addr_e1, strip 1 odd cols at addr_o1 (= addr_e1 + 2).
template <std::uint32_t addr_e0, std::uint32_t addr_o0, std::uint32_t addr_e1, std::uint32_t addr_o1>
inline void _sinkhorn_strip_pair_multi_(std::uint32_t iters)
{
    // ── Load 4 LREGs (2 strips × 2 parities). ──────────────────────────
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, addr_e0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, addr_o0);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, addr_e1);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, addr_o1);

    for (std::uint32_t i = 0; i < iters; ++i)
    {
        _sinkhorn_strip_pair_body_();
    }

    // ── Round LREG0..3 (fp32) → bf16 (RNE) before store. ───────────────
    // SFPSTORE → bf16 dst defaults to TRUNCATION (round-toward-zero).
    // Apply RNE here so the one-and-only bf16 conversion per strip-pair
    // is unbiased.
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;

    // ── Store. ─────────────────────────────────────────────────────────
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, addr_e0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, addr_o0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, addr_e1);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, addr_o1);
}

template <std::uint32_t NUM_FACES_USED, std::uint32_t ITERS, std::uint32_t EPS_BITS, bool SINGLE_SUBMAT, std::uint32_t VALID_H, std::uint32_t VALID_W>
inline void _sinkhorn_4x4_()
{
    // Row-norm + col-norm in SFPU, configurable over 1..4 faces.
    //
    // Each face has 4 row strips of 4 rows × 16 cols each. We process them
    // as 2 strip-pairs (LREG0..3 hold one pair, LREG4..7 are scratch; the
    // lane-parity mask and eps live in the config registers LREG11/LREG13).
    //
    // Face 0 base addrs: e0=0, o0=2, e1=4, o1=6, e2=8, o2=10, e3=12, o3=14.
    // Face 1 base addrs: face 0 + 16.
    // Face 2 base addrs: face 0 + 32.
    // Face 3 base addrs: face 0 + 48.
    //
    // NUM_FACES_USED controls which faces to process:
    //   1 = only face 0 (hc <= 16)
    //   2 = faces 0, 1
    //   4 = faces 0, 1, 2, 3 (full pack-64)
    //
    // ITERS is threaded into each strip-pair so the row+col norm runs
    // ITERS times in fp32 lregs without bf16 round-tripping between iters.

    _sinkhorn_zero_rect_padding_<VALID_H, VALID_W>();

    // Program LREG11 (parity mask) and LREG13 (eps) for this tile. Must run
    // after exp_tile (exp_tile_init reprograms LREG12..14 each tile).
    _sinkhorn_program_constants_<EPS_BITS>();

    // Face 0, strips 0+1 (rows 0..7).
    _sinkhorn_strip_pair_multi_<0, 2, 4, 6>(ITERS);
    // comb_from_mixes packs exactly one 4x4 matrix at the top-left of face 0.
    // Its remaining strips are padding, so avoid normalizing rows 8..15.
    if constexpr (!SINGLE_SUBMAT)
    {
        // Face 0, strips 2+3 (rows 8..15).
        _sinkhorn_strip_pair_multi_<8, 10, 12, 14>(ITERS);
    }

    if constexpr (NUM_FACES_USED >= 2)
    {
        // Face 1, strips 0+1 (rows 0..7, cols 16..31).
        _sinkhorn_strip_pair_multi_<16, 18, 20, 22>(ITERS);
        // Face 1, strips 2+3.
        _sinkhorn_strip_pair_multi_<24, 26, 28, 30>(ITERS);
    }

    if constexpr (NUM_FACES_USED >= 3)
    {
        // Face 2, strips 0+1 (rows 16..23, cols 0..15).
        _sinkhorn_strip_pair_multi_<32, 34, 36, 38>(ITERS);
        // Face 2, strips 2+3.
        _sinkhorn_strip_pair_multi_<40, 42, 44, 46>(ITERS);
    }

    if constexpr (NUM_FACES_USED >= 4)
    {
        // Face 3, strips 0+1 (rows 16..23, cols 16..31).
        _sinkhorn_strip_pair_multi_<48, 50, 52, 54>(ITERS);
        // Face 3, strips 2+3.
        _sinkhorn_strip_pair_multi_<56, 58, 60, 62>(ITERS);
    }
}

} // namespace sfpu

} // namespace ckernel
