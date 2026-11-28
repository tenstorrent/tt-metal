// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// int32 -> bfloat16 compute variants (perf experiment int32_bf16_fused_sfpu). Same block loop / CB contract
// as kernels/baseline_compute.cpp (a copy of the op's compute kernel); only the SFPU element(s) differ.
//
// VARIANT (define):
//   1 fused_plain   : ONE custom SFPU element (Int32ToBf16FusedPlain), ONE init for the whole kernel. Per DEST row:
//                     int32 -> fp32 RNE (abs, SFPCAST, setsgn, INT32_MIN fixup) then fp32 -> bf16 RNE
//                     (x + 0x7fff + lsb) & 0xffff0000, stored as FP32 (packer truncation of the zeroed low half
//                     is then exact). 13 SFPU instructions / row, no SFPLOADMACRO.
//   2 twopass       : ONE custom SFPU element (Int32ToBf16TwoPass), ONE init: the LLK int32->fp32 SFPLOADMACRO pass
//                     (4 cycles / row) followed by an RNE rounding pass over the same tile. The LLK fp32->fp16b
//                     round reads its "1" from LREG12, which the int32->fp32 macro needs as -31 (SFPSHFT2 shift),
//                     so the rounding pass here reads "1" from LREG5 (loaded per call); 0x7fff / 0xffff0000 stay in
//                     LREG13 / LREG14 (unused by the macro). That removes the constant-register conflict that forces
//                     the two-element chain to re-init both inits on every tile.
//   3 packer_round  : Typecast<Int32, Float32> only; the packer narrows the fp32 DEST to bf16. Exact only if the
//                     Blackhole packer rounds fp32 -> bf16 RNE (measured, not assumed).
//   4 direct_llk    : Typecast<Int32, Float16_b> (stock's LLK arm; SFP_STOCH_RND rounds ties half-away). Timing
//                     reference only; known inexact.
//   5 veltkamp_plain      : fused, int32->fp32 as 1 + fp32->bf16 RNE by Veltkamp splitting (3 MADs, sigma = 65537).
//   6 twopass_veltkamp    : LLK int32->fp32 SFPLOADMACRO pass + plain Veltkamp pass (5 instr / row).
//   7 twopass_veltkamp_x2 : as 6, two rows per iteration interleaved (MAD latency hidden).
//   8 veltkamp_fused_x2   : fused single pass, two rows interleaved, LLK-style L[L7] INT32_MIN fixup (10 instr / row).
//   9 twopass_veltkamp_macro (RECOMMENDED): LLK int32->fp32 SFPLOADMACRO pass (4 instr / row) + an SFPLOADMACRO
//                     Veltkamp pass (3 instr / row: LM, G, H; D + store scheduled by the macro). One init configures
//                     both passes' macros (macro 0 / templates 0-1 = LLK, macros 1-2 / templates 2-3 = rounding).
//
// Veltkamp exactness: host-verified over ALL 2^32 int32 inputs (numpy float32 RNE); device-verified bit-exact on
// ~6.5M values (two seeds, witnesses, INT32_MIN/MAX, ties around 2^24..2^31).
//
// Raw LLK note (helper bypass): the kernel_lib eltwise chain is still used for the CB/DEST/pack walk; what the
// helper cannot express is an SFPU function the LLK library does not ship (an RNE int32->bf16 in one SFPU element)
// nor a way to keep two Typecast elements' inits hoisted when their programmable-constant registers collide.
// The custom elements below are UnaryOp CRTP elements (chain-compatible) whose bodies are raw TTI SFPU code.

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "g6_typecast_zones.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace exp_sfpu {
using namespace ckernel;
using namespace ckernel::sfpu;

// fp32 -> bf16 RNE constants: LREG13 = 0x7fff, LREG14 = 0xffff0000. Fused-plain also uses LREG12 = 1.
inline void init_fused_plain() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <int ITERATIONS>
inline void calculate_fused_plain() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                   // L1 = |x| (INT32_MIN stays 0x80000000)
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);                     // L2 = fp32(L1), sign-magnitude, RNE
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                // L0 = sign(x) | expman(L2)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc = (x == INT32_MIN)
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);                            // L0 += -2^31 (-0.0 -> -2^31)
        TTI_SFPENCC(0, 0, 0, 0);
        // fp32 -> bf16 round-to-nearest-even on the bit pattern.
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, 5);                // L3 = L0 >> 16
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);                            // L3 &= 1 (lsb of bf16)
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);  // L0 += 0x7fff
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);   // L0 += lsb
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG0, 0);                            // L0 &= 0xffff0000
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

// Two-pass: the LLK int32->fp32 loadmacro init (LREG12 = -31, templates, macro 0), plus the round constants in
// LREG13 / LREG14.
inline void init_twopass() {
    init_typecast_int32_to_fp32<false>();
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <int ITERATIONS>
inline void calculate_round_rne_lreg5() {
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 1);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);
        TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

// Veltkamp splitting (s = 16): with sigma = 2^16 + 1 and RNE fp32 arithmetic,
//   g = RN(sigma * x), d = RN(x - g), h = RN(g + d)
// is x rounded to 24 - 16 = 8 significant bits, i.e. fp32 -> bf16 RNE. Host-verified exhaustively over all 2^32
// int32 inputs (x = fp32 RNE(n); numpy float32 = IEEE RNE). Three MADs instead of five integer ops.
inline void init_veltkamp_plain() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm1 = 65537.0f;  // LREG13 = sigma
}

template <int ITERATIONS>
inline void calculate_veltkamp_fused_plain() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);    // g = x * sigma
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG3, 0);  // d = x - g
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);     // h = g + d
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

inline void init_twopass_veltkamp() {
    init_typecast_int32_to_fp32<false>();
    sfpi::vConstFloatPrgm1 = 65537.0f;  // LREG13 = sigma (the int32->fp32 macro only needs LREG12)
}

template <int ITERATIONS>
inline void calculate_round_veltkamp() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);
        TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

// Two rows per iteration (rows a = base, b = base + 2): independent MAD chains interleaved so each MAD's
// result latency is covered by the other row's instruction. Stores: ADDR_MOD_6 (+2) twice -> base += 4.
template <int ITERATIONS>
inline void calculate_round_veltkamp_x2() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_7, 2);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);    // ga
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);    // gb
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // da
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_neg1, p_sfpu::LREG4, p_sfpu::LREG4, 0);  // db
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);     // ha
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);     // hb
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

// Fused, two rows per iteration. int32 -> fp32 exactly as the LLK (abs, sign bit of |v| -> L7, SFPCAST RNE,
// setsgn, MAD with VA = L[L7] in {L0 = 0.0, L1 = -2^31} for the INT32_MIN fixup), then Veltkamp.
inline void init_veltkamp_fused_x2() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    preload_sign_magnitude_cast_fixup();  // LREG12 = -31
    sfpi::vConstFloatPrgm1 = 65537.0f;    // LREG13 = sigma
}

template <int ITERATIONS>
inline void calculate_veltkamp_fused_x2() {
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // -2**31
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
        TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPABS(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // L7 = |a| >> 31
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG6, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
        TTI_SFPMAD(0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 4);  // xa = L[L7] * 1 + xa
        TTI_SFPSHFT2(p_sfpu::LREG5, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // L7 = |b| >> 31
        TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
        TTI_SFPMAD(0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 4);                 // xb = L[L7] * 1 + xb
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);    // ga
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);    // gb
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // da
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_neg1, p_sfpu::LREG4, p_sfpu::LREG4, 0);  // db
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);     // ha
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);     // hb
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

// SFPLOADMACRO Veltkamp rounding pass (runs after the LLK int32->fp32 macro pass on the same tile).
// Encoding per unit byte (see ckernel_sfpu_mul_int32.h): bit7 VB=VD (else VC=VD), bit6 VD=L16, bits5-3 delay,
// bits2-0 template 4+i or 3 = store. Uses macros 1, 2 and templates 2, 3 (the int32->fp32 pass owns macro 0 and
// templates 0, 1), so both passes' configuration lives side by side after ONE init.
//
// Row r: x_r = L[r % 4] (loaded by the macro), g_r = L[4 + r % 2]. Stream per row: LM_r, G_r, H_{r-1}.
//   G_r : g_r = x_r * sigma                  (explicit MAD)
//   D_r : [x_r] = g_r * -1 + x_r              (macro MAD, template 2 + r % 2, delay 2)
//   H_r : x_r = x_r * 1 + g_r                 (explicit MAD, one row later)
//   S_r : store [x_r] FP32                    (macro store, delay 7)
// All units count delay in issued instructions (misc 0xf00), switched per pass and restored for the LLK pass.
#ifndef VK_CONFIG_SETTLE_NOPS
#define VK_CONFIG_SETTLE_NOPS 3
#endif
constexpr std::uint32_t VK_MISC_ROUND = 0xf00 | InstrModLoadStore::FP32;
constexpr std::uint32_t VK_MISC_INT32_TO_FP32 = 0x100 | InstrModLoadStore::FP32;

inline void init_twopass_veltkamp_macro() {
    init_typecast_int32_to_fp32<false>();
    sfpi::vConstFloatPrgm1 = 65537.0f;  // LREG13 = sigma
    // Templates 2 / 3: D with g = L4 / L5.
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, 0, 12 + 2, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_neg1, 0, 12 + 3, 0);
    constexpr std::uint32_t disabled = 7 << 3;
    constexpr std::uint32_t store_bits = (7 << 3) | 3;
    {
        constexpr std::uint32_t mad_bits = (2 << 3) | (4 + 2);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        constexpr std::uint32_t mad_bits = (2 << 3) | (4 + 3);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
}

template <int ITERATIONS>
inline void calculate_round_veltkamp_macro() {
    TTI_SFPCONFIG(VK_MISC_ROUND, 8, 1);
    for (int i = 0; i < VK_CONFIG_SETTLE_NOPS; i++) {
        TTI_SFPNOP;
    }
#pragma GCC unroll 8
    for (int r = 0; r < ITERATIONS; r++) {
        const int x = r & 3;
        const int g = 4 + (r & 1);
        const int m = 1 + (r & 1);
        TT_SFPLOADMACRO((m << 2) | x, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TT_SFPMAD(x, p_sfpu::LREG13, p_sfpu::LCONST_0, g, 0);
        if (r > 0) {
            const int xp = (r - 1) & 3;
            const int gp = 4 + ((r - 1) & 1);
            TT_SFPMAD(xp, p_sfpu::LCONST_1, gp, xp, 0);
        } else {
            TTI_SFPNOP;
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    {
        constexpr int xl = (ITERATIONS - 1) & 3;
        constexpr int gl = 4 + ((ITERATIONS - 1) & 1);
        TTI_SFPMAD(xl, p_sfpu::LCONST_1, gl, xl, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPCONFIG(VK_MISC_INT32_TO_FP32, 8, 1);
}
}  // namespace exp_sfpu
#endif

namespace compute_kernel_lib {

template <Dst Slot = Dst::D0>
struct Int32ToBf16FusedPlain : UnaryOp<Int32ToBf16FusedPlain<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_fused_plain)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_fused_plain<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16TwoPass : UnaryOp<Int32ToBf16TwoPass<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_twopass)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<false, 8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_round_rne_lreg5<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16VeltkampPlain : UnaryOp<Int32ToBf16VeltkampPlain<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_veltkamp_plain)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_veltkamp_fused_plain<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16TwoPassVeltkamp : UnaryOp<Int32ToBf16TwoPassVeltkamp<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_twopass_veltkamp)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<false, 8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_round_veltkamp<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16TwoPassVeltkampX2 : UnaryOp<Int32ToBf16TwoPassVeltkampX2<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_twopass_veltkamp)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<false, 8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_round_veltkamp_x2<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16VeltkampFusedX2 : UnaryOp<Int32ToBf16VeltkampFusedX2<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_veltkamp_fused_x2)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_veltkamp_fused_x2<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

template <Dst Slot = Dst::D0>
struct Int32ToBf16TwoPassVeltkampMacro : UnaryOp<Int32ToBf16TwoPassVeltkampMacro<Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(exp_sfpu::init_twopass_veltkamp_macro)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<false, 8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_sfpu::calculate_round_veltkamp_macro<8>, to_u32(Slot) + slot_offset, VectorMode::RC)));
    }
};

}  // namespace compute_kernel_lib

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;
    constexpr uint32_t df_int32 = static_cast<uint32_t>(DataFormat::Int32);
    constexpr uint32_t df_fp32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr uint32_t df_bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
    constexpr InputSpec in_spec = input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block);
    constexpr OutputSpec out_spec = output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd);
    constexpr InputSpec in_spec_hoisted = input(
        cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block, DataFormatReconfig::Disabled);
    constexpr OutputSpec out_spec_hoisted =
        output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled);

    DataflowBuffer dfb_in(cb_input_tiles);
    DataflowBuffer dfb_out(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    auto sfpu_elem = []() {
        if constexpr (VARIANT == 1) {
            return Int32ToBf16FusedPlain<>{};
        } else if constexpr (VARIANT == 2) {
            return Int32ToBf16TwoPass<>{};
        } else if constexpr (VARIANT == 5) {
            return Int32ToBf16VeltkampPlain<>{};
        } else if constexpr (VARIANT == 6) {
            return Int32ToBf16TwoPassVeltkamp<>{};
        } else if constexpr (VARIANT == 7) {
            return Int32ToBf16TwoPassVeltkampX2<>{};
        } else if constexpr (VARIANT == 8) {
            return Int32ToBf16VeltkampFusedX2<>{};
        } else if constexpr (VARIANT == 9) {
            return Int32ToBf16TwoPassVeltkampMacro<>{};
        } else if constexpr (VARIANT == 3) {
            return Typecast<df_int32, df_fp32>{};
        } else {
            return Typecast<df_int32, df_bf16>{};
        }
    };

    G6_ZONE("CONVERT");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done < block_tiles) ? (core_tiles - done) : block_tiles;
        const auto shape = IterationShape::tiles(block_extent).block_size(block_extent);
        if (done == 0) {
            eltwise_chain<InitReconfigOwner::Chain>(
                shape, CopyTile<in_spec>{dfb_in}, sfpu_elem(), PackTile<out_spec>{dfb_out});
        } else {
            eltwise_chain<InitReconfigOwner::Caller>(
                shape, CopyTile<in_spec_hoisted>{dfb_in}, sfpu_elem(), PackTile<out_spec_hoisted>{dfb_out});
        }
    }
}
