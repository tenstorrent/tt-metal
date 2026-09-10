// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU RoPE for [1, 32] tiny tiles.
//
// Addressing (BlackholeA0 ISA, SFPLOAD):
//     Row    = (Addr & ~3) + Lane/8
//     Column = (Lane & 7) * 2 + (Addr & 2 ? 1 : 0)
// so one vector is 4 rows x 8 columns of ONE column parity, Addr>>2 selects the
// row group, and bit 1 selects even/odd columns.  For a [1, 32] tile only DEST
// row 0 of each face is live, so lanes 0..7 carry the data and lanes 8..31 read
// the zeros custom_mm's clear_src left in rows 1..3.  Those lanes are computed
// and stored back too, but a [1, 32] tile only ever packs row 0, so they are
// inert.
//
// cos/sin are in the interleaved (Meta) layout, each angle duplicated across
// the two slots of its pair: cos = (c0,c0,c1,c1,...).  A single even-parity
// load therefore serves BOTH x_even and x_odd.

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

namespace rope
{
// DEST rows per Tile32x32 slot (copy_tile / pack_tile addressing) and per face.
constexpr std::uint32_t TILE_ROWS = 64;
constexpr std::uint32_t FACE_ROWS = 16;
// Mod1 bit that negates the VA operand of SFPMAD (VD = -(VA*VB) + VC).
constexpr std::uint32_t NEGATE_VA = 1;

constexpr std::uint8_t ZERO_ADDR_MOD = ADDR_MOD_7;
} // namespace rope

inline void sfpu_rope_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(rope::ZERO_ADDR_MOD);
}

inline void sfpu_rope_dest_setup()
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

/**
 * cos/sin for one face into LREG0/LREG1.  Every head shares them at a given width
 * tile, so the load is hoisted out of the head loop.  A single even-parity load
 * serves both x parities (see the layout note at the top).
 */
inline void sfpu_rope_load_cos_sin(const std::uint32_t cos_addr, const std::uint32_t sin_addr)
{
    constexpr std::uint32_t FMT = static_cast<std::uint32_t>(InstrModLoadStore::FP16B);
    TT_SFPLOAD(p_sfpu::LREG0, FMT, rope::ZERO_ADDR_MOD, cos_addr);
    TT_SFPLOAD(p_sfpu::LREG1, FMT, rope::ZERO_ADDR_MOD, sin_addr);
}

/**
 * Scales the cos/sin in LREG0/LREG1 by ``scale_fp32``, an fp32 bit pattern.
 */
inline void sfpu_rope_scale_cos_sin(const std::uint32_t scale_fp32)
{
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, scale_fp32 & 0xFFFFu);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, scale_fp32 >> 16);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
}

/**
 * One face: 8 complex pairs of a [1, 32] tile, rotated by the cos/sin already in
 * LREG0/LREG1.
 *
 *   x'_even = cos*x_even - sin*x_odd
 *   x'_odd  = sin*x_even + cos*x_odd
 *
 * x_addr is the absolute DEST row of the face's first row, and must be 4-row aligned
 * (the caller's x_base/x_stride asserts guarantee it).  Only the address-bearing
 * instructions take the runtime TT_ form; the multiplies keep their inline encoding.
 */
inline void sfpu_rope_face(const std::uint32_t x_addr)
{
    constexpr std::uint32_t FMT = static_cast<std::uint32_t>(InstrModLoadStore::FP16B);
    constexpr std::uint8_t AM   = rope::ZERO_ADDR_MOD;

    TT_SFPLOAD(p_sfpu::LREG4, FMT, AM, x_addr);     // x_even
    TT_SFPLOAD(p_sfpu::LREG5, FMT, AM, x_addr + 2); // x_odd

    // No SFPNOPs: hardware auto-stalls a dependent SFPMAD read, and both products
    // are computed before either is consumed, so each pair is already a cycle apart.
    // LREG2 = cos*x_even ; LREG3 = sin*x_even
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    // LREG6 = LREG2 - sin*x_odd ; LREG7 = LREG3 + cos*x_odd
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, rope::NEGATE_VA);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG3, p_sfpu::LREG7, 0);

    TT_SFPSTORE(p_sfpu::LREG6, FMT, AM, x_addr);
    TT_SFPSTORE(p_sfpu::LREG7, FMT, AM, x_addr + 2);
}

/**
 * Ht*Wt x-tiles starting at DEST row ``x_base``, ``x_stride`` rows apart, against
 * Wt cos tiles at ``cos_base`` and Wt sin tiles at ``sin_base`` (``cs_stride``
 * apart). cos/sin are shared by every head — decode rotates all heads at one
 * position — so the nesting is (width tile, face) outer and head inner, and each
 * cos/sin pair is loaded once rather than per tile.
 *
 * x_stride is 64 when x came from copy_tile (Tile32x32 slots) and 32 when it is
 * a custom_mm<dense_packing> result still sitting in DEST.
 *
 * scale_fp32 is required rather than defaulted: under has_scale a missing value
 * would scale cos/sin by zero and silently zero every output. Callers with
 * has_scale=false pass 0.
 */
template <
    std::uint32_t Ht,
    std::uint32_t Wt,
    std::uint32_t x_base,
    std::uint32_t x_stride,
    std::uint32_t cos_base,
    std::uint32_t sin_base,
    std::uint32_t cs_stride,
    bool has_scale = false>
inline void sfpu_rope_all_rows(const std::uint32_t scale_fp32)
{
    constexpr std::uint32_t F = rope::FACE_ROWS;

    static_assert((F & 3) == 0, "face stride must keep rows 4-row aligned");
    static_assert((x_base & 3) == 0 && (x_stride & 3) == 0, "x rows must be 4-row aligned");
    static_assert((cos_base & 3) == 0 && (sin_base & 3) == 0 && (cs_stride & 3) == 0, "cos/sin rows must be 4-row aligned");

    constexpr std::uint32_t head_stride = Wt * x_stride;

    for (std::uint32_t w = 0; w < Wt; w++)
    {
        for (std::uint32_t f = 0; f < 2; f++)
        {
            const std::uint32_t cs_off = w * cs_stride + f * F;
            sfpu_rope_load_cos_sin(cos_base + cs_off, sin_base + cs_off);
            if constexpr (has_scale)
            {
                // Amortized over the Ht heads that reuse this cos/sin pair.
                sfpu_rope_scale_cos_sin(scale_fp32);
            }
            std::uint32_t x_addr = x_base + w * x_stride + f * F;
            for (std::uint32_t h = 0; h < Ht; h++)
            {
                sfpu_rope_face(x_addr);
                x_addr += head_stride;
            }
        }
    }
}

} // namespace sfpu
} // namespace ckernel
