// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU LLK for the per-tile forward-substitution triangle solve  L X = RHS  (unit lower-triangular
// L). Blackhole-only: the SFPMAD accumulation chain relies on the HW scoreboard.
//
// One DST-register tile input, one DST-register tile output. L does NOT occupy a DST register — it
// is read element-by-element straight from L1 (via broadcast_tile_value_bf16). L is the plain unit
// lower-triangular tile (diagonal an implicit 1, strict-lower entries NOT negated); the per-column
// update subtracts L[row][col] * X[col] directly via an SFPMAD that negates the product:
//   dst_in  : the right-hand-side tile (RHS)
//   dst_out : receives the solution X of  L X = RHS
//   l1_tri_base : L1 byte address of element (0,0) of the unit-lower-tri L tile
//
// HW-validated against torch.linalg.solve_triangular on Blackhole (PCC 0.9999).

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

// Broadcast one bf16 element of an L1-resident tile to all 32 lanes of an SFPU register.
//
// Instead of extracting the value out of DEST (which would need cross-lane SFPU shuffles across the
// 4x8 lane grid), read the 16-bit bf16 word straight from the tile in L1 and hand its raw bits to
// SFPLOADI, which splats a 16-bit immediate to all 32 lanes for free. bf16 is the upper 16 bits of
// fp32, so SFPLOADI in FLOATB mode reconstructs the exact fp32 value in every lane.
//
//   l1_tile_base : byte address in L1 of element (0,0) of the tile (bf16, standard TILE layout).
//                  Obtain via CircularBuffer::get_tile_address(tile_index), which resolves the read
//                  pointer on the UNPACK thread and mailboxes the byte address to MATH.
//   x, y         : logical (row, col) in the 32x32 tile, each 0..31.
//   out_lreg     : destination SFPU register index (p_sfpu::LREGn).
//
// Runs on the MATH thread (same thread as SFPLOADI). The tile must already be resident in L1
// (cb_wait_front done by the caller) before this is called.
inline void broadcast_tile_value_bf16(std::uint32_t l1_tile_base, std::uint32_t x, std::uint32_t y, std::uint32_t out_lreg)
{
    // Standard bf16 TILE layout: a 32x32 tile is 4 faces of 16x16 stored [f0, f1, f2, f3], row-major
    // within each face. face = (x/16)*2 + (y/16); uint16 element index = face*256 + (x%16)*16 + (y%16).
    const std::uint32_t face               = ((x >> 4) << 1) + (y >> 4);
    const std::uint32_t elem               = (face << 8) + ((x & 15) << 4) + (y & 15);
    volatile tt_l1_ptr std::uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(l1_tile_base);
    const std::uint16_t bits               = tile[elem];
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
inline std::uint32_t triangle_solve_row_offset(std::uint32_t row)
{
    const std::uint32_t chunk = row >> 2;
    const std::uint32_t r     = row & 3u;
    return (chunk & 3u) * 4u + (chunk >> 2) * 32u + (r & 1u) * 2u + (r >> 1) * 16u;
}

template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void triangle_solve(const std::uint32_t dst_in, const std::uint32_t dst_out, const std::uint32_t l1_tri_base)
{
    // Each tile occupies 64 rows in DEST.
    constexpr std::uint32_t dst_tile_size = 64;

    // Logical dimension of the tile (32x32).
    constexpr std::uint32_t TILE_DIM = 32;

    // Walk the STRICTLY-lower triangle of the unit lower-triangular tile L, whose element (0,0) lives
    // at byte address `l1_tri_base` in L1. (L no longer occupies a DEST register — it is read straight
    // from L1 here — so the op takes a single DEST input, the RHS, and a single DEST output.)
    //
    // The 32 rows are walked in 8 chunks of 4 so the row loop exposes a natural every-4th-row
    // boundary (the start of each chunk, r == 0):
    //   chunk loop : which block of 4 rows        (0 .. 7)
    //   row loop   : row within the chunk         (0 .. 3)  -> row = chunk * 4 + r
    //   col loop   : column, from the left edge up to — but not including — the diagonal
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

        for (std::uint32_t r = 0; r < ROW_CHUNK; r++)
        {
            const std::uint32_t row = chunk * ROW_CHUNK + r;

            // Accumulator LREG for this row. After the block transpose, row `row` lives in LREG
            // (row % 4) — i.e. 0,1,2,3 repeating within each chunk — and currently holds RHS[row].
            // Forward substitution turns it into X[row] in place before we store it.
            const std::uint32_t out_lreg = (chunk * ROW_CHUNK + r) % 4u;

            // Forward substitution: X[row] = RHS[row] - sum_{col < row} L[row][col] * X[col].
            // The subtraction is performed directly by the SFPMAD below: it negates the product
            // (SFPMAD_MOD1_NEGATE_VA) so each column update does out -= L[row][col] * X[col] using
            // the plain (non-negated) L values read from L1.
            // Each already-solved X[col] (col < row, computed on an earlier row) sits in dst_out at
            // its per-row slot triangle_solve_row_offset(col); re-read it and fold it in. dst_out
            // holds the rows in row-oriented "scratch" form during the solve; the final transpose
            // corrects the whole tile in place at the end.
            for (std::uint32_t col = 0; col < row; col++)
            {
                // L[row][col] -> LREG7 (splat across all lanes; leaves LREG0..3 rows intact).
                broadcast_tile_value_bf16(l1_tri_base, row, col, p_sfpu::LREG7);
                // X[col] -> LREG4 from dst_out (single load == inverse of the store, so it comes back
                // row-oriented regardless of the block scramble).
                //
                // NOP #1 (SFPLOADI -> SFPLOAD sequencing): broadcast_tile_value_bf16 ends in a
                // runtime-emitted TT_SFPLOADI, and the very next instruction is another SFPU load-class
                // op (SFPLOAD). On the Blackhole SFPU these back-to-back load-class ops must not issue in
                // adjacent slots; a single SFPNOP separates them. Established during HW bring-up
                // (the end-to-end solve validates at PCC 0.9999 with these NOPs in place).
                TTI_SFPNOP;
                TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, dst_out * dst_tile_size + triangle_solve_row_offset(col));
                // out_lreg = out_lreg - (LREG7 * LREG4)   (dest == src_c == the row's accumulator).
                // mod1 = SFPMAD_MOD1_NEGATE_VA (1) negates the product before the add.
                // Runtime LREG index, so this is TT_SFPMAD (not TTI_).
                //
                // NOP #2 (SFPLOAD -> SFPMAD load-use hazard): LREG4 is written by the SFPLOAD above and
                // read as an operand by this SFPMAD. The load result is not available to an immediately
                // following dependent op, so one SFPNOP covers the load-to-use latency on Blackhole.
                // Removing it lets the MAD read a stale LREG4 and silently corrupts the accumulation.
                TTI_SFPNOP;
                TT_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, out_lreg, out_lreg, 1); // SFPMAD_MOD1_NEGATE_VA
            }

            // Stash the solved row in dst_out at its per-row {0,2,16,18} slot (same in-tile layout as
            // the input load), row-oriented. A single load re-reads it row-oriented on later rows, and
            // the final in-place transpose corrects the whole tile to standard layout at the end.
            // Runtime LREG index -> TT_SFPSTORE.
            const std::uint32_t out_addr = dst_out * dst_tile_size + triangle_solve_row_offset(row);
            TT_SFPSTORE(out_lreg, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_addr);
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
        const std::uint32_t group_base = dst_out * dst_tile_size + (chunk & 3u) * 4u + (chunk >> 2) * 32u;

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
