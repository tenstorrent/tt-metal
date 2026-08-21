// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU LLK for the per-tile forward-substitution triangle solve  L X = RHS  (unit lower-triangular
// L). Blackhole-only: the SFPMAD accumulation chain relies on the HW scoreboard.
//
// PROTOTYPE NOTE (local, do not merge): this is the OPTIMIZED microcode posted by fvranicTT in
// PR #53437 review comment id 3802921292 (~1368 issued SFPU instructions vs 2632 in the PR head).
// CONTRACT DIFFERENCE vs the PR head (6bad73c): L must be supplied PRE-NEGATED (strict-lower
// entries negated offline; SFPMAD mod1=0 accumulates L_neg[row][col]*X[col] directly). The PR
// head instead takes plain L and negates in the MAD (mod1=1). GDN's negN tile is exactly the
// pre-negated input this version wants. HW-unvalidated by the PR author.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

// Pointer to one bf16 element of an L1-resident tile.
//
//   tile     : base pointer (element (0,0)) of the tile (bf16, standard TILE layout).
//   row, col : logical (row, col) in the 32x32 tile, each 0..31.
//
// Standard bf16 TILE layout: a 32x32 tile is 4 faces of 16x16 stored [f0, f1, f2, f3], row-major
// within each face. face = (row/16)*2 + (col/16); uint16 element index = face*256 + (row%16)*16 +
// (col%16). Consecutive columns in the same face are adjacent; consecutive rows are stride 16.
inline volatile tt_l1_ptr std::uint16_t* triangle_l1_elem_ptr(volatile tt_l1_ptr std::uint16_t* tile, const std::uint32_t row, const std::uint32_t col)
{
    const std::uint32_t face = ((row >> 4) << 1) + (col >> 4);
    const std::uint32_t elem = (face << 8) + ((row & 15) << 4) + (col & 15);
    return tile + elem;
}

inline void broadcast_tile_value_bf16(std::uint32_t l1_tile_base, std::uint32_t x, std::uint32_t y, std::uint32_t out_lreg)
{
    volatile tt_l1_ptr std::uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(l1_tile_base);
    const std::uint16_t bits               = *triangle_l1_elem_ptr(tile, x, y);
    // FLOATB: interpret the 16-bit immediate as bf16 and expand it to fp32 across all 32 lanes.
    TT_SFPLOADI(out_lreg, sfpi::SFPLOADI_MOD0_FLOATB, bits);
}

// DEST address offset where the solve stashes logical row `row` (0..31) of the output.
//
// A single SFPSTORE writes a 4-row x 8-col block, so each row gets its own block slot, and the 32
// rows are laid out in the SAME face/parity blocks the reshuffle (and input load) use. That way the
// reshuffle can read 4 rows per block at group_base + {0,2,16,18}, transpose, and emit the standard
// tile. For row = chunk*4 + r:
//   group_base = (chunk&3)*4 + (chunk>>2)*32   (4-row group within a face-pair, +32 to next pair)
//   slot(r)    = (r&1)*2 + (r>>1)*16           (r=0,1 -> first two of face-1 {0,2};
//                                               r=2,3 -> first two of face-2 {16,18})
inline constexpr std::uint32_t kTriangleRowOff[32] = {
    // chunk 0..3 (rows 0..15): group_base = (chunk&3)*4
    0,
    2,
    16,
    18,
    4,
    6,
    20,
    22,
    8,
    10,
    24,
    26,
    12,
    14,
    28,
    30,
    // chunk 4..7 (rows 16..31): group_base = (chunk&3)*4 + 32
    32,
    34,
    48,
    50,
    36,
    38,
    52,
    54,
    40,
    42,
    56,
    58,
    44,
    46,
    60,
    62};

inline constexpr std::uint32_t triangle_solve_row_offset(std::uint32_t row)
{
    return kTriangleRowOff[row];
}

// Right-looking rank-1 update of the live 4-row window: X[col] (LREG4) times L[row0+r][col]
// (stride-16 in L1) folded into LREG0..3. L1 loads of rows 1..3 are issued in the shadow of the
// previous SFPLOADI / SFPMAD.
inline void triangle_apply_prev_col(volatile tt_l1_ptr std::uint16_t* l_col, const std::uint32_t x_addr)
{
    const std::uint16_t b0 = l_col[0];
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, x_addr);
    const std::uint16_t b1 = l_col[16];
    // Bring-up: SFPLOAD and SFPLOADI are both load-class and must not issue in adjacent slots.
    // The SFPLOADI also sits between SFPLOAD and the first SFPMAD, covering the original
    // SFPLOAD -> SFPMAD load-use of LREG4. Later MADs in this column consume LREG4 after that
    // gap and consume LREG7 from an SFPLOADI (1-cycle, next-insn-ready); no further NOPs.
    TTI_SFPNOP;
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b0);
    const std::uint16_t b2 = l_col[32];
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG0, 0);

    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b1);
    const std::uint16_t b3 = l_col[48];
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG1, 0);

    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b2);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG2, 0);

    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b3);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG3, 0);
}

template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void triangle_solve(const std::uint32_t dst_in, const std::uint32_t dst_out, const std::uint32_t l1_tri_base)
{
    // Each tile occupies 64 rows in DEST.
    constexpr std::uint32_t dst_tile_size = 64;

    // Logical dimension of the tile (32x32).
    constexpr std::uint32_t TILE_DIM = 32;

    volatile tt_l1_ptr std::uint16_t* const tile = reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(l1_tri_base);
    const std::uint32_t dst_out_base             = dst_out * dst_tile_size;

    // Walk the STRICTLY-lower triangle of the (pre-negated) unit lower-triangular tile L, whose
    // element (0,0) lives at byte address `l1_tri_base` in L1. (L no longer occupies a DEST
    // register — it is read straight from L1 here — so the op takes a single DEST input, the RHS,
    // and a single DEST output.)
    //
    // The 32 rows are walked in 8 chunks of 4. After the block transpose, LREG0..3 hold the four
    // RHS rows of the chunk and become the in-place accumulators X[row0+r]:
    //   Phase 1 (previous columns): for each col < row0, load X[col] once and MAD it into all 4 rows.
    //   Phase 2 (within chunk):     row r uses X already sitting in LREG0..r-1; then store the row.
    // The unit diagonal (row == col) is skipped; only entries with col < row participate.
    constexpr std::uint32_t ROW_CHUNK  = 4;
    constexpr std::uint32_t NUM_CHUNKS = TILE_DIM / ROW_CHUNK; // 8
    for (std::uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++)
    {
        // Bring this chunk's 4 rows of the RHS tile (dst_in) into LREG0..3 in row-major form,
        // mirroring Welford's `_welfords_load_block_`: bracket the 4 loads with a transpose before
        // and after so the 4-LREG x lane block is flipped and LREG0..3 end up holding 4 whole rows.
        //
        // Same offset layout Welford uses: base = (I*32) + (4*J) with I = chunk>>2 (face-column
        // half, +32 jump) and J = chunk&3 (4-row group within the half, +4 step); the {0,2,16,18}
        // within-block offsets pick the block's 4 rows across the two stacked faces. Loads use the
        // SRCB format mode, as Welford does.
        const std::uint32_t group_base = dst_in * dst_tile_size + (chunk & 3u) * 4u + (chunk >> 2) * 32u;
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 2);
        TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 16);
        TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 18);
        TTI_SFPTRANSP(0, 0, 0, 0);
        // LREG0..3 now hold rows (chunk*4 + 0..3) of the RHS.

        const std::uint32_t row0 = chunk * ROW_CHUNK;

        // Phase 1: previous-chunk columns. L[row0][col] walks +1 inside a face; the four rows of the
        // chunk are stride 16. Columns 0..15 and 16..row0-1 live in different faces, so the pointer
        // is reset at col 16 rather than crossing the 16-wide face boundary.
        //
        // Forward substitution: X[row] = RHS[row] - sum_{col < row} L[row][col] * X[col].
        // The subtraction is folded into an ADD because the L tile is negated offline, so runtime
        // just accumulates L_neg[row][col] * X[col].
        if (row0 > 0)
        {
            const std::uint32_t nface0                = row0 < 16u ? row0 : 16u;
            volatile tt_l1_ptr std::uint16_t* l_face0 = triangle_l1_elem_ptr(tile, row0, 0);
            for (std::uint32_t col = 0; col < nface0; col++)
            {
                triangle_apply_prev_col(l_face0 + col, dst_out_base + kTriangleRowOff[col]);
            }
            if (row0 > 16u)
            {
                volatile tt_l1_ptr std::uint16_t* l_face1 = triangle_l1_elem_ptr(tile, row0, 16);
                for (std::uint32_t col = 16; col < row0; col++)
                {
                    triangle_apply_prev_col(l_face1 + (col - 16u), dst_out_base + kTriangleRowOff[col]);
                }
            }
        }

        // Phase 2: within-chunk lower triangle. Row r of the chunk depends on X[row0+k] for k < r,
        // which are still live in LREG k. Then stash the solved row in dst_out at its per-row
        // {0,2,16,18} slot, row-oriented. dst_out address is runtime -> TT_SFPSTORE.
        // r = 0: no strictly-lower entries in this chunk.
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, dst_out_base + kTriangleRowOff[row0 + 0]);

        {
            volatile tt_l1_ptr std::uint16_t* l_row = triangle_l1_elem_ptr(tile, row0 + 1, row0);
            const std::uint16_t b0                  = l_row[0];
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b0);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, 0);
            TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, dst_out_base + kTriangleRowOff[row0 + 1]);
        }

        {
            volatile tt_l1_ptr std::uint16_t* l_row = triangle_l1_elem_ptr(tile, row0 + 2, row0);
            const std::uint16_t b0                  = l_row[0];
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b0);
            const std::uint16_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, 0);
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b1);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
            TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, dst_out_base + kTriangleRowOff[row0 + 2]);
        }

        {
            volatile tt_l1_ptr std::uint16_t* l_row = triangle_l1_elem_ptr(tile, row0 + 3, row0);
            const std::uint16_t b0                  = l_row[0];
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b0);
            const std::uint16_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b1);
            const std::uint16_t b2 = l_row[2];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, b2);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, dst_out_base + kTriangleRowOff[row0 + 3]);
        }
    }

    // -----------------------------------------------------------------------------------------
    // Output transpose (in place on dst_out): the solve stashed each row into dst_out row-oriented at
    // its {0,2,16,18} slot. Read the 4 rows of each block back, SFPTRANSP them into the split-face
    // layout, and store the standard tile back to the SAME slots.
    //
    // A single SFPLOAD/SFPSTORE moves a 4-row x 8-column block; the address decodes as
    // bits[9:2]=4-row group, bit[1]=even/odd column half, bit[0]=unused. The 4 rows of a block sit at
    // group_base + {0, 2, 16, 18} (face-1 even/odd, face-2 even/odd), with
    // group_base = (chunk&3)*4 + (chunk>>2)*32. Each block's 4 loads complete before its 4 stores, and
    // blocks are disjoint, so the in-place transpose never clobbers data it has not yet read.
    // -----------------------------------------------------------------------------------------
    for (std::uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++)
    {
        const std::uint32_t group_base = dst_out_base + (chunk & 3u) * 4u + (chunk >> 2) * 32u;

        TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 2);
        TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 16);
        TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 18);

        TTI_SFPTRANSP(0, 0, 0, 0);

        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 0);
        TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 2);
        TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 16);
        TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, group_base + 18);
    }
}

inline void triangle_solve_init()
{
    // No SFPU state to program for the scaffold.
}

} // namespace sfpu
} // namespace ckernel
