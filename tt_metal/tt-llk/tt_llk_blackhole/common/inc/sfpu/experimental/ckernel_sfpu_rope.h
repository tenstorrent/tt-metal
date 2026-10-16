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
#include <type_traits>
#include <utility>

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

// LREGs of the load-macro body. e' must be LREG0: the store x_even's macro schedules reads the register its own VD field
// names, which is 0. x_even and x_odd stay below LREG4 so the load macro's VD high bit, which doubles as address bit 0,
// stays 0.
constexpr std::uint32_t M_X_EVEN = p_sfpu::LREG1;
constexpr std::uint32_t M_X_ODD  = p_sfpu::LREG2;
constexpr std::uint32_t M_P      = p_sfpu::LREG3;
constexpr std::uint32_t M_COS    = p_sfpu::LREG4;
constexpr std::uint32_t M_SIN    = p_sfpu::LREG5;
constexpr std::uint32_t M_SCALE  = p_sfpu::LREG6;

template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t load_store_mod()
{
    return static_cast<std::uint32_t>(is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B);
}

// Passes i to f as a std::integral_constant so it stays a constant expression.
template <typename F, std::uint32_t... I>
inline void unroll(F&& f, std::integer_sequence<std::uint32_t, I...>)
{
    (f(std::integral_constant<std::uint32_t, I> {}), ...);
}

template <std::uint32_t N, typename F>
inline void unroll(F&& f)
{
    unroll(f, std::make_integer_sequence<std::uint32_t, N> {});
}
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

/**
 * @note Call sfpu_rope_all_rows or sfpu_rope_fused_all_rows right after it: their first instruction is the third of
 *       the three SFPLOAD needs after an FPU DEST write.
 */
inline void sfpu_rope_dest_setup()
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

/**
 * cos/sin for one face into LREG0/LREG1.  Every head shares them at a given width
 * tile, so the load is hoisted out of the head loop.  A single even-parity load
 * serves both x parities (see the layout note at the top).
 */
template <bool is_fp32_dest_acc_en = false>
inline void sfpu_rope_load_cos_sin(const std::uint32_t cos_addr, const std::uint32_t sin_addr)
{
    // BF16 inputs can be unpacked into FP32 DEST: use the register representation.
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    TT_SFPLOAD(p_sfpu::LREG0, FMT, rope::ZERO_ADDR_MOD, cos_addr);
    TT_SFPLOAD(p_sfpu::LREG1, FMT, rope::ZERO_ADDR_MOD, sin_addr);
}

/**
 * Loads ``scale_fp32``, an fp32 bit pattern, into LREG2; sfpu_rope_mul_cos_sin scales the cos/sin in LREG0/LREG1 by it.
 */
inline void sfpu_rope_load_scale(const std::uint32_t scale_fp32)
{
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, scale_fp32 & 0xFFFFu);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, scale_fp32 >> 16);
}

inline void sfpu_rope_mul_cos_sin()
{
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
}

inline void sfpu_rope_rotate()
{
    // No SFPNOPs: hardware auto-stalls a dependent SFPMAD read, and both products
    // are computed before either is consumed, so each pair is already a cycle apart.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, rope::NEGATE_VA);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG3, p_sfpu::LREG7, 0);
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
template <bool is_fp32_dest_acc_en = false>
inline void sfpu_rope_face(const std::uint32_t x_addr)
{
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    constexpr std::uint8_t AM   = rope::ZERO_ADDR_MOD;

    TT_SFPLOAD(p_sfpu::LREG4, FMT, AM, x_addr);     // x_even
    TT_SFPLOAD(p_sfpu::LREG5, FMT, AM, x_addr + 2); // x_odd
    sfpu_rope_rotate();
    TT_SFPSTORE(p_sfpu::LREG6, FMT, AM, x_addr);
    TT_SFPSTORE(p_sfpu::LREG7, FMT, AM, x_addr + 2);
}

template <std::uint32_t cos_addr, std::uint32_t sin_addr, bool is_fp32_dest_acc_en>
inline void sfpu_rope_load_cos_sin_imm()
{
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    TTI_SFPLOAD(p_sfpu::LREG0, FMT, rope::ZERO_ADDR_MOD, cos_addr);
    TTI_SFPLOAD(p_sfpu::LREG1, FMT, rope::ZERO_ADDR_MOD, sin_addr);
}

template <std::uint32_t x_addr, bool is_fp32_dest_acc_en>
inline void sfpu_rope_face_imm()
{
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    constexpr std::uint8_t AM   = rope::ZERO_ADDR_MOD;

    TTI_SFPLOAD(p_sfpu::LREG4, FMT, AM, x_addr);
    TTI_SFPLOAD(p_sfpu::LREG5, FMT, AM, x_addr + 2);
    sfpu_rope_rotate();
    TTI_SFPSTORE(p_sfpu::LREG6, FMT, AM, x_addr);
    TTI_SFPSTORE(p_sfpu::LREG7, FMT, AM, x_addr + 2);
}

/**
 * Programs the two load macros of the rope body. x_even's SFPLOADMACRO also issues q = sin * x_even into its own LREG
 * and, four instructions later, the store of e' from LREG0; x_odd's issues o' = cos * x_odd + q into the staging LREG
 * and its store. A face then takes four issue slots (two macro loads, p and e') instead of eight, with every MAD the
 * plain body does. Delays count issued SFPU instructions, so issue stalls cannot reorder them. The macros stay
 * programmed after the call: an op whose init programs its own needs that init again before its next call.
 */
inline void sfpu_rope_macro_setup()
{
    // InstructionTemplate[0] and [1] (VD 12 and 13); the load supplies VB and VD.
    TTI_SFPMAD(rope::M_SIN, p_sfpu::LREG0, p_sfpu::LCONST_0, 12, 0);
    TTI_SFPMAD(rope::M_COS, p_sfpu::LREG0, rope::M_X_EVEN, 13, 0);
    // One byte per unit: bit 7 load LREG as VB, bit 6 staging LREG, bits 5:3 delay, bits 2:0 instruction (3 store,
    // 4 and 5 the templates).
    constexpr std::uint32_t q_mad   = 0x80 | (1 << 3) | 4;
    constexpr std::uint32_t e_store = 0x80 | (4 << 3) | 3;
    constexpr std::uint32_t o_mad   = 0x80 | 0x40 | (1 << 3) | 5;
    constexpr std::uint32_t o_store = 0x40 | (3 << 3) | 3;
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, q_mad << 8);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, e_store << 8);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, o_mad << 8);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, o_store << 8);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    // Stores take the load's format; the MAD and store units wait on issued instructions.
    TTI_SFPCONFIG(0xA30, 8, 1);
}

template <std::uint32_t cs_addr, bool has_scale, bool is_fp32_dest_acc_en>
inline void sfpu_rope_load_cos_sin_macro()
{
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    // cos first: the previous group's last o' reads it in this slot.
    TTI_SFPLOAD(rope::M_COS, FMT, rope::ZERO_ADDR_MOD, cs_addr);
    TTI_SFPLOAD(rope::M_SIN, FMT, rope::ZERO_ADDR_MOD, cs_addr + 2);
    if constexpr (has_scale)
    {
        TTI_SFPMAD(rope::M_COS, rope::M_SCALE, p_sfpu::LCONST_0, rope::M_COS, 0);
        TTI_SFPMAD(rope::M_SIN, rope::M_SCALE, p_sfpu::LCONST_0, rope::M_SIN, 0);
    }
}

template <std::uint32_t x_addr, bool is_fp32_dest_acc_en>
inline void sfpu_rope_face_macro()
{
    constexpr std::uint32_t FMT = rope::load_store_mod<is_fp32_dest_acc_en>();
    constexpr std::uint8_t AM   = rope::ZERO_ADDR_MOD;

    TTI_SFPLOADMACRO((0 << 2) | rope::M_X_EVEN, FMT, AM, x_addr);
    TTI_SFPMAD(rope::M_COS, rope::M_X_EVEN, p_sfpu::LCONST_0, rope::M_P, 0);
    TTI_SFPLOADMACRO((1 << 2) | rope::M_X_ODD, FMT, AM, x_addr + 2);
    TTI_SFPMAD(rope::M_SIN, rope::M_X_ODD, rope::M_P, p_sfpu::LREG0, rope::NEGATE_VA);
}

template <
    std::uint32_t Ht,
    std::uint32_t Wt,
    std::uint32_t x_base,
    std::uint32_t x_stride,
    std::uint32_t cos_base,
    std::uint32_t sin_base,
    std::uint32_t cs_stride,
    bool has_scale>
inline void sfpu_rope_all_rows_macro(const std::uint32_t scale_fp32)
{
    constexpr std::uint32_t F           = rope::FACE_ROWS;
    constexpr std::uint32_t FMT         = rope::load_store_mod<false>();
    constexpr std::uint8_t AM           = rope::ZERO_ADDR_MOD;
    constexpr std::uint32_t head_stride = Wt * x_stride;

    sfpu_rope_macro_setup();
    if constexpr (has_scale)
    {
        TT_SFPLOADI(rope::M_SCALE, sfpi::SFPLOADI_MOD0_LOWER, scale_fp32 & 0xFFFFu);
        TT_SFPLOADI(rope::M_SCALE, sfpi::SFPLOADI_MOD0_UPPER, scale_fp32 >> 16);
    }
    for (std::uint32_t w = 0; w < Wt; w++)
    {
        for (std::uint32_t f = 0; f < 2; f++)
        {
            const std::uint32_t cs_off = w * cs_stride + f * F;
            TT_SFPLOAD(rope::M_COS, FMT, AM, cos_base + cs_off);
            TT_SFPLOAD(rope::M_SIN, FMT, AM, sin_base + cs_off);
            if constexpr (has_scale)
            {
                TTI_SFPMAD(rope::M_COS, rope::M_SCALE, p_sfpu::LCONST_0, rope::M_COS, 0);
                TTI_SFPMAD(rope::M_SIN, rope::M_SCALE, p_sfpu::LCONST_0, rope::M_SIN, 0);
            }
            std::uint32_t x_addr = x_base + w * x_stride + f * F;
            for (std::uint32_t h = 0; h < Ht; h++)
            {
                TT_SFPLOADMACRO((0 << 2) | rope::M_X_EVEN, FMT, AM, x_addr);
                TTI_SFPMAD(rope::M_COS, rope::M_X_EVEN, p_sfpu::LCONST_0, rope::M_P, 0);
                TT_SFPLOADMACRO((1 << 2) | rope::M_X_ODD, FMT, AM, x_addr + 2);
                TTI_SFPMAD(rope::M_SIN, rope::M_X_ODD, rope::M_P, p_sfpu::LREG0, rope::NEGATE_VA);
                x_addr += head_stride;
            }
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <
    std::uint32_t Ht,
    std::uint32_t Wt,
    std::uint32_t x_base,
    std::uint32_t x_stride,
    std::uint32_t cs_base,
    std::uint32_t cs_stride,
    bool has_scale,
    std::uint32_t tile_h,
    bool cos_sin_per_row,
    bool is_fp32_dest_acc_en>
inline void sfpu_rope_fused_all_rows_macro(const std::uint32_t scale_fp32)
{
    constexpr std::uint32_t F                  = rope::FACE_ROWS;
    constexpr std::uint32_t head_stride        = Wt * x_stride;
    constexpr std::uint32_t num_row_halves     = (tile_h > 16) ? 2 : 1;
    constexpr std::uint32_t live_rows_per_half = (tile_h > 16) ? 16 : tile_h;
    constexpr std::uint32_t num_row_groups     = (live_rows_per_half + 3) / 4;

    sfpu_rope_macro_setup();
    if constexpr (has_scale)
    {
        TT_SFPLOADI(rope::M_SCALE, sfpi::SFPLOADI_MOD0_LOWER, scale_fp32 & 0xFFFFu);
        TT_SFPLOADI(rope::M_SCALE, sfpi::SFPLOADI_MOD0_UPPER, scale_fp32 >> 16);
    }
    rope::unroll<Wt>(
        [&](auto w_index)
        {
            constexpr std::uint32_t w = decltype(w_index)::value;
            rope::unroll<2>(
                [&](auto f_index)
                {
                    constexpr std::uint32_t f = decltype(f_index)::value;
                    if constexpr (!cos_sin_per_row)
                    {
                        sfpu_rope_load_cos_sin_macro<cs_base + w * cs_stride + f * F, has_scale, is_fp32_dest_acc_en>();
                    }
                    rope::unroll<num_row_halves>(
                        [&](auto half_index)
                        {
                            rope::unroll<num_row_groups>(
                                [&](auto rg_index)
                                {
                                    constexpr std::uint32_t row = decltype(half_index)::value * 2 * F + f * F + decltype(rg_index)::value * 4;
                                    if constexpr (cos_sin_per_row)
                                    {
                                        sfpu_rope_load_cos_sin_macro<cs_base + w * cs_stride + row, has_scale, is_fp32_dest_acc_en>();
                                    }
                                    rope::unroll<Ht>(
                                        [&](auto h_index)
                                        {
                                            constexpr std::uint32_t x_addr = x_base + w * x_stride + row + decltype(h_index)::value * head_stride;
                                            sfpu_rope_face_macro<x_addr, is_fp32_dest_acc_en>();
                                        });
                                });
                        });
                });
        });
    // The last face's o' and both of its stores land on these.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
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

#ifndef DISABLE_SFPLOADMACRO
    if constexpr (Ht * Wt * 2 >= 8)
    {
        sfpu_rope_all_rows_macro<Ht, Wt, x_base, x_stride, cos_base, sin_base, cs_stride, has_scale>(scale_fp32);
        return;
    }
#endif
    if constexpr (!has_scale)
    {
        // The third slot after sfpu_rope_dest_setup; with a scale, its SFPLOADIs take it.
        TTI_NOP;
    }
    for (std::uint32_t w = 0; w < Wt; w++)
    {
        for (std::uint32_t f = 0; f < 2; f++)
        {
            const std::uint32_t cs_off = w * cs_stride + f * F;
            if constexpr (has_scale)
            {
                sfpu_rope_load_scale(scale_fp32);
            }
            sfpu_rope_load_cos_sin(cos_base + cs_off, sin_base + cs_off);
            if constexpr (has_scale)
            {
                // Amortized over the Ht heads that reuse this cos/sin pair.
                sfpu_rope_mul_cos_sin();
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

/**
 * Ht*Wt x-tiles starting at DEST row ``x_base``, ``x_stride`` rows apart, against
 * Wt fused cos/sin tiles at ``cs_base`` (``cs_stride`` apart). The fused tiles
 * are shared by every head — decode rotates all heads at one position — so the
 * nesting is (width tile, column-face) outer and head inner, and each fused
 * tile is loaded once rather than per tile.
 *
 * ``tile_h`` is the live row count of each x tile (1, 2, 4, 8, 16, or 32).
 * When ``cos_sin_per_row`` is false (single-token decode) one fused row is
 * reused for every row-group of that column-face; the fused tile must then
 * have that row replicated across live rows if ``tile_h > 1``.  When true
 * (block decode) each 4-row group loads its matching fused rows.
 *
 * ``is_fp32_dest_acc_en`` selects the DEST register representation for both
 * x and fused phases, independently of their original input memory format.
 *
 * x_stride is 64 when x came from copy_tile (Tile32x32 slots) and 32 when it is
 * a custom_mm<dense_packing> result still sitting in DEST.
 */
template <
    std::uint32_t Ht,
    std::uint32_t Wt,
    std::uint32_t x_base,
    std::uint32_t x_stride,
    std::uint32_t cs_base,
    std::uint32_t cs_stride,
    bool has_scale           = false,
    std::uint32_t tile_h     = 1,
    bool cos_sin_per_row     = false,
    bool is_fp32_dest_acc_en = false>
inline void sfpu_rope_fused_all_rows(const std::uint32_t scale_fp32)
{
    constexpr std::uint32_t F = rope::FACE_ROWS;

    static_assert(tile_h == 1 || tile_h == 2 || tile_h == 4 || tile_h == 8 || tile_h == 16 || tile_h == 32, "rope_sfpu: tile_h must be 1, 2, 4, 8, 16, or 32");
    static_assert((F & 3) == 0, "face stride must keep rows 4-row aligned");
    static_assert((x_base & 3) == 0 && (x_stride & 3) == 0, "x rows must be 4-row aligned");
    static_assert((cs_base & 3) == 0 && (cs_stride & 3) == 0, "cos/sin rows must be 4-row aligned");

    constexpr std::uint32_t head_stride        = Wt * x_stride;
    constexpr std::uint32_t num_row_halves     = (tile_h > 16) ? 2 : 1;
    constexpr std::uint32_t live_rows_per_half = (tile_h > 16) ? 16 : tile_h;
    constexpr std::uint32_t num_row_groups     = (live_rows_per_half + 3) / 4;

#ifndef DISABLE_SFPLOADMACRO
    // Setup and drain cost 12 issue slots per call and each face saves 4, so from eight faces a call saves 20 or more.
    if constexpr (Ht * Wt * 2 * num_row_halves * num_row_groups >= 8)
    {
        sfpu_rope_fused_all_rows_macro<Ht, Wt, x_base, x_stride, cs_base, cs_stride, has_scale, tile_h, cos_sin_per_row, is_fp32_dest_acc_en>(scale_fp32);
        return;
    }
#endif
    if constexpr (!has_scale)
    {
        TTI_NOP;
    }
    rope::unroll<Wt>(
        [&](auto w_index)
        {
            constexpr std::uint32_t w = decltype(w_index)::value;
            rope::unroll<2>(
                [&](auto f_index)
                {
                    constexpr std::uint32_t f = decltype(f_index)::value;
                    if constexpr (!cos_sin_per_row)
                    {
                        constexpr std::uint32_t cs_addr = cs_base + w * cs_stride + f * F;
                        if constexpr (has_scale)
                        {
                            sfpu_rope_load_scale(scale_fp32);
                        }
                        sfpu_rope_load_cos_sin_imm<cs_addr, cs_addr + 2 /* sin_addr */, is_fp32_dest_acc_en>();
                        if constexpr (has_scale)
                        {
                            sfpu_rope_mul_cos_sin();
                        }
                    }
                    rope::unroll<num_row_halves>(
                        [&](auto half_index)
                        {
                            rope::unroll<num_row_groups>(
                                [&](auto rg_index)
                                {
                                    constexpr std::uint32_t row = decltype(half_index)::value * 2 * F + f * F + decltype(rg_index)::value * 4;
                                    if constexpr (cos_sin_per_row)
                                    {
                                        constexpr std::uint32_t cs_addr = cs_base + w * cs_stride + row;
                                        if constexpr (has_scale)
                                        {
                                            sfpu_rope_load_scale(scale_fp32);
                                        }
                                        sfpu_rope_load_cos_sin_imm<cs_addr, cs_addr + 2 /* sin_addr */, is_fp32_dest_acc_en>();
                                        if constexpr (has_scale)
                                        {
                                            sfpu_rope_mul_cos_sin();
                                        }
                                    }
                                    rope::unroll<Ht>(
                                        [&](auto h_index)
                                        {
                                            constexpr std::uint32_t x_addr = x_base + w * x_stride + row + decltype(h_index)::value * head_stride;
                                            sfpu_rope_face_imm<x_addr, is_fp32_dest_acc_en>();
                                        });
                                });
                        });
                });
        });
}

} // namespace sfpu
} // namespace ckernel
