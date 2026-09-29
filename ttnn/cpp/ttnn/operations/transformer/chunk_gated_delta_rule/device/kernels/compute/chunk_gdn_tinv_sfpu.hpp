// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SFPU forward-substitution solve for the GDN WY inverse, T_inv = (I - negN)^-1, on ONE 32x32 tile.
// Blackhole only (the SFPMAD accumulation chain relies on the Blackhole scoreboard).
//
//   X[r] = RHS[r] + sum_{c < r} negN[r][c] * X[c]      (unit diagonal implicit, never read)
//
// negN is the PRE-NEGATED strictly-lower factor prep already builds (cb.scr3 = -strictly_lower(N)), so
// the MAD accumulates negN[r][c] * X[c] directly. With RHS = I the solution is T_inv. negN is read
// element-wise, in place, from its fp32 L1 tile: two SFPLOADIs (UPPER, LOWER) splat each element across
// the 32 SFPU lanes, so L is never staged or rounded. The RHS rows and the solution live in DEST (fp32
// when fp32_dest_acc_en, as in the GDN kernels).
//
// The instruction schedule is the optimized microcode Filip Vranic posted on the SFPU triangle-solve LLK
// (tt-metal #53437 by Vasisht Suresh, review comment r3802921292): rows are solved in 8 chunks of 4 held in
// LREG0..3 — a right-looking rank-1 update from every previous column, then the small in-chunk triangle —
// with the solved rows stashed row-oriented in the output tile and transposed back at the end. It is kept
// op-local until a generic triangle-solve primitive lands in the compute API.

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if defined(ARCH_BLACKHOLE)
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"
#endif
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
namespace sfpu {

// Element (row, col) of a standard 32x32 TILE-layout tile: 4 row-major 16x16 faces [f0, f1, f2, f3].
inline constexpr std::uint32_t gdn_tinv_elem(std::uint32_t row, std::uint32_t col) {
    return ((((row >> 4) << 1) + (col >> 4)) << 8) + ((row & 15) << 4) + (col & 15);
}

// DEST offset (within a 64-row tile) where the solve stashes logical row `row` of the output: a single
// SFPSTORE writes a 4-row x 8-col block, so each row gets its own block slot, laid out in the same
// face/parity blocks the input load and the final transpose use (group_base + {0, 2, 16, 18}).
inline constexpr std::uint32_t kGdnTinvRowOff[32] = {0,  2,  16, 18, 4,  6,  20, 22, 8,  10, 24, 26, 12, 14, 28, 30,
                                                     32, 34, 48, 50, 36, 38, 52, 54, 40, 42, 56, 58, 44, 46, 60, 62};

// kNeg (P3_FLAPREP): the solve reads kk*L_mask (prep's scr1) instead of negN and forms negN's element in
// flight. prep's negN = diag(kk*L) - kk*L unpacked kk*L through srcB, which truncates fp32 to tf32 (the low 13
// mantissa bits), so its strictly-lower element is exactly -tf32(x): flip the sign bit, clear bits 12..0.
// The UPPER/LOWER SFPLOADI words of that value: imm16 = hi16 ^ 0x8000 and lo16 & 0xE000. The opcode base has
// a zero imm16 field, so base + imm == base | imm == base ^ imm: each word is one shift/mask plus one xor/or.
template <bool kNeg>
constexpr std::uint32_t gdn_tinv_hi_word(std::uint32_t base, std::uint32_t bits) {
    return kNeg ? ((bits >> 16) ^ (base ^ 0x8000u)) : (base + (bits >> 16));
}
template <bool kNeg>
constexpr std::uint32_t gdn_tinv_lo_word(std::uint32_t base, std::uint32_t bits) {
    return kNeg ? ((bits & 0xE000u) | base) : (base + (bits & 0xFFFFu));
}

// Splat one fp32 L element into LREG7 across all lanes.
template <bool kNeg = false>
inline void gdn_tinv_load_l(std::uint32_t bits) {
    ckernel::instrn_buffer[0] =
        gdn_tinv_hi_word<kNeg>(TT_OP_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0), bits);
    ckernel::instrn_buffer[0] =
        gdn_tinv_lo_word<kNeg>(TT_OP_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 0), bits);
}

// Previous-column rank-1 updates of the live 4-row window (P3_FLAPREP): X[col] times L[row0 + r][col] (stride 16
// in L1) folded into LREG0..3. The SFPMADs (operands, order) are those of the original one-column helper, so the
// solve is bit-identical, but the instruction stream is cheaper (T_inv 6.1 -> 5.2 us on the math thread):
//  * the four L elements of a column splat into two registers (LREG6/LREG7) so each pair of SFPLOADIs is
//    followed by two independent MADs;
//  * X[col] is double-buffered in LREG4/LREG5: X[col + 1] loads between the column's third and fourth MAD
//    (never next to a load-class SFPLOADI), which removes the per-column SFPNOP;
//  * the eight L1 reads of a column pair issue first, ahead of the instruction words that use them, and the
//    SFPLOADI/SFPLOAD words are built from constant opcodes (one shift or zero-extend plus one add each).
#define GDN_TINV_COL(XA, NEXT_X_WORD, B0, B1, B2, B3)               \
    ib[0] = gdn_tinv_hi_word<kNeg>(kHi6, B0);                       \
    ib[0] = gdn_tinv_lo_word<kNeg>(kLo6, B0);                       \
    ib[0] = gdn_tinv_hi_word<kNeg>(kHi7, B1);                       \
    ib[0] = gdn_tinv_lo_word<kNeg>(kLo7, B1);                       \
    TTI_SFPMAD(p_sfpu::LREG6, XA, p_sfpu::LREG0, p_sfpu::LREG0, 0); \
    TTI_SFPMAD(p_sfpu::LREG7, XA, p_sfpu::LREG1, p_sfpu::LREG1, 0); \
    ib[0] = gdn_tinv_hi_word<kNeg>(kHi6, B2);                       \
    ib[0] = gdn_tinv_lo_word<kNeg>(kLo6, B2);                       \
    ib[0] = gdn_tinv_hi_word<kNeg>(kHi7, B3);                       \
    ib[0] = gdn_tinv_lo_word<kNeg>(kLo7, B3);                       \
    TTI_SFPMAD(p_sfpu::LREG6, XA, p_sfpu::LREG2, p_sfpu::LREG2, 0); \
    ib[0] = (NEXT_X_WORD);                                          \
    TTI_SFPMAD(p_sfpu::LREG7, XA, p_sfpu::LREG3, p_sfpu::LREG3, 0);

// Columns [c_begin, c_end) of the window at rows row0..row0+3; l = &L[row0][c_begin] (4 rows at stride 16).
// c_end - c_begin is even (row0 is a multiple of 4) and X[c_begin] is already in LREG4 (c_begin even).
// out_base = DEST row of the solution tile (the stashed rows X[c] live at out_base + kGdnTinvRowOff[c]).
template <bool kNeg>
inline __attribute__((always_inline)) void gdn_tinv_prev_cols(
    volatile tt_l1_ptr std::uint32_t* l, std::uint32_t c_begin, std::uint32_t c_end, std::uint32_t out_base) {
    volatile std::uint32_t* const ib = &ckernel::instrn_buffer[0];
    constexpr std::uint32_t kHi6 = TT_OP_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0);
    constexpr std::uint32_t kLo6 = TT_OP_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0);
    constexpr std::uint32_t kHi7 = TT_OP_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0);
    constexpr std::uint32_t kLo7 = TT_OP_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 0);
    const std::uint32_t ld4 = TT_OP_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 0) + out_base;
    const std::uint32_t ld5 = TT_OP_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 0) + out_base;
#pragma GCC unroll 1
    for (std::uint32_t c = c_begin; c < c_end; c += 2) {
        const std::uint32_t a0 = l[0], a1 = l[16], a2 = l[32], a3 = l[48];
        const std::uint32_t e0 = l[1], e1 = l[17], e2 = l[33], e3 = l[49];
        l += 2;
        // c + 2 <= 30 here; at the window's last column it loads a not-yet-solved row into the idle buffer.
        GDN_TINV_COL(p_sfpu::LREG4, ld5 + kGdnTinvRowOff[c + 1], a0, a1, a2, a3)
        GDN_TINV_COL(p_sfpu::LREG5, ld4 + kGdnTinvRowOff[c + 2], e0, e1, e2, e3)
    }
}
#undef GDN_TINV_COL

template <bool kNeg>
inline void gdn_tinv_trisolve(std::uint32_t dst_in, std::uint32_t dst_out, std::uint32_t l1_base) {
    constexpr std::uint32_t dst_tile_size = 64;  // DEST rows per tile
    volatile tt_l1_ptr std::uint32_t* const tile = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(l1_base);
    const std::uint32_t out_base = dst_out * dst_tile_size;

    for (std::uint32_t chunk = 0; chunk < 8; chunk++) {
        // The chunk's 4 RHS rows into LREG0..3, row-major (transpose, 4 block loads, transpose).
        const std::uint32_t in_base = dst_in * dst_tile_size + (chunk & 3u) * 4u + (chunk >> 2) * 32u;
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, in_base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, in_base + 2);
        TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, in_base + 16);
        TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, in_base + 18);
        TTI_SFPTRANSP(0, 0, 0, 0);

        const std::uint32_t row0 = chunk * 4;
        // Previous-chunk columns. Columns 0..15 and 16..row0-1 live in different faces, so the element
        // pointer restarts at column 16 instead of walking across the face boundary.
        if (row0 > 0) {
            TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + kGdnTinvRowOff[0]);
            TTI_SFPNOP;  // SFPLOAD and SFPLOADI must not issue in adjacent slots
            const std::uint32_t nface0 = row0 < 16u ? row0 : 16u;
            gdn_tinv_prev_cols<kNeg>(tile + gdn_tinv_elem(row0, 0), 0, nface0, out_base);
            if (row0 > 16u) {
                gdn_tinv_prev_cols<kNeg>(tile + gdn_tinv_elem(row0, 16), 16, row0, out_base);
            }
        }

        // In-chunk triangle: row r uses X[row0 + k], k < r, still live in LREG k; then stash the row.
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + kGdnTinvRowOff[row0 + 0]);
        {
            volatile tt_l1_ptr std::uint32_t* l_row = tile + gdn_tinv_elem(row0 + 1, row0);
            gdn_tinv_load_l<kNeg>(l_row[0]);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, 0);
            TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + kGdnTinvRowOff[row0 + 1]);
        }
        {
            volatile tt_l1_ptr std::uint32_t* l_row = tile + gdn_tinv_elem(row0 + 2, row0);
            const std::uint32_t b0 = l_row[0];
            gdn_tinv_load_l<kNeg>(b0);
            const std::uint32_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, 0);
            gdn_tinv_load_l<kNeg>(b1);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
            TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + kGdnTinvRowOff[row0 + 2]);
        }
        {
            volatile tt_l1_ptr std::uint32_t* l_row = tile + gdn_tinv_elem(row0 + 3, row0);
            const std::uint32_t b0 = l_row[0];
            gdn_tinv_load_l<kNeg>(b0);
            const std::uint32_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            gdn_tinv_load_l<kNeg>(b1);
            const std::uint32_t b2 = l_row[2];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            gdn_tinv_load_l<kNeg>(b2);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + kGdnTinvRowOff[row0 + 3]);
        }
    }

    // Output transpose, in place: read each block's 4 row-oriented rows back, SFPTRANSP them into the
    // standard face layout and store them to the same slots (loads of a block precede its stores and
    // blocks are disjoint, so nothing is clobbered before it is read).
    for (std::uint32_t chunk = 0; chunk < 8; chunk++) {
        const std::uint32_t base = out_base + (chunk & 3u) * 4u + (chunk >> 2) * 32u;
        TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + 2);
        TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + 16);
        TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + 18);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + 0);
        TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + 2);
        TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + 16);
        TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + 18);
    }
}

inline void gdn_tinv_trisolve_init() {}

}  // namespace sfpu
#endif  // TRISC_MATH

// Solve (I - negN) X = DST[idst_in] into DST[idst_out]. negN is tile `l_tile_idx` of cb_l (fp32,
// front-waited, read in place). DST must be acquired; call gdn_tinv_trisolve_tile_init() first.
// kNeg: cb_l holds kk*L_mask instead of negN (see gdn_tinv_hi_word); the solve is bit-identical either way.
template <bool kNeg = false>
ALWI void gdn_tinv_trisolve_tile(CircularBuffer& cb_l, uint32_t l_tile_idx, uint32_t idst_in, uint32_t idst_out) {
    // UNPACK resolves the tile's L1 address and mailboxes it to MATH and PACK.
    const uint32_t l1_base = cb_l.get_tile_address(l_tile_idx);
    // The packer wrote this tile, and the MATH RISC reads it through its (write-through) L1 cache: the tile
    // sits at the same CB address every chunk, so without an invalidate a cached line can be the previous
    // chunk's L.
    MATH((invalidate_l1_cache()));
    MATH((_llk_math_eltwise_sfpu_start_(0)));  // the solve addresses DEST absolutely (idst * 64 rows)
    MATH((sfpu::gdn_tinv_trisolve<kNeg>(idst_in, idst_out, l1_base)));
    MATH((_llk_math_eltwise_sfpu_done_()));
}

ALWI void gdn_tinv_trisolve_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::gdn_tinv_trisolve_init))); }

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
