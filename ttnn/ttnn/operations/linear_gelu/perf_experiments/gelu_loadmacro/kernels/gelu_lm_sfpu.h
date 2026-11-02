// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Raw-LLK tanh-GELU candidates (Blackhole). Bypasses the sfpi-compiled calculate_gelu_tanh_fast (the helper
// path behind GeluTanhActivation<1>): that body is one serial dependency chain per 32-lane row, 34 issued
// instructions + ~10 SFPMAD auto-stall cycles (~43.6 cycles/row measured). The mechanism exploited here is
// SFPU issue packing: two DEST rows are processed in lockstep so every SFPMAD's 2-cycle latency is covered by
// the partner row's instruction, constants live in LRegs (shared by both rows) instead of per-row SFPLOADIs,
// and the reciprocal refinement is fused with the final multiply by x.
//
// Same function as the baseline: x / (1 + 2^(x (A + B x^2))) with the exp_21f bit trick, the same degree-4
// 2^f polynomial and the same [0, 254] clamp. Only the reciprocal refinement differs: one cubic step
// r (1 + e + e^2), e = 1 - d r, instead of two Newton steps (both are at fp32 rounding level from SFPARECIP's
// ~8-bit seed), fused as res = xr + xr (e + e^2) with xr = x r. V3+ split xlog2 with SFPSTOCHRND; V4+ move the
// per-row tail onto SFPLOADMACRO; V6 carries the series to e^3 (fp32-unbiased like the baseline).
//
// SFPU state owned by gelu_lm_init (must not be reprogrammed by anything else on the issuing thread between
// tiles): LReg11 = 254 (replaces vConstNeg1), LReg12-14 constants, LoadMacro InstructionTemplate[0..1],
// Sequence[0..1] and Misc (V4+), ADDR_MOD_6 = dest += 4 (V5+). Blackhole only (relies on SFPMAD auto-stall,
// SFPMAD NEGATE_VA and BH SFPSTOCHRND / SFPLOADMACRO encodings); assumes fp32 DEST (no bf16 RNE before store).
#pragma once

#include "ckernel_sfpu_gelu.h"

namespace ckernel::sfpu {

namespace gelu_lm {

constexpr std::uint32_t L_ZERO = p_sfpu::LCONST_0;  // 9
constexpr std::uint32_t L_ONE = p_sfpu::LCONST_1;   // 10
constexpr std::uint32_t L_254 = 11;                 // programmable constant, set by gelu_lm_init
constexpr std::uint32_t L_A = 12;                   // vConstFloatPrgm0 = GELU_TANH_FAST_A
constexpr std::uint32_t L_B = 13;                   // vConstFloatPrgm1 = GELU_TANH_FAST_B
constexpr std::uint32_t L_C1 = 14;                  // vConstFloatPrgm2 = GELU_TANH_FAST_EXP2_C1

constexpr std::uint16_t BF16_127 = 0x42FE;
constexpr std::uint16_t BF16_254 = 0x437E;
constexpr std::uint16_t BF16_C2 = 0x2777;  // GELU_TANH_FAST_EXP2_C2 (bf16-exact)
constexpr std::uint16_t BF16_C3 = 0x1AD6;  // GELU_TANH_FAST_EXP2_C3
constexpr std::uint16_t BF16_C4 = 0x0E5C;  // GELU_TANH_FAST_EXP2_C4

constexpr std::uint32_t NEG_A = 1;  // SFPMAD_MOD1_NEGATE_VA

// Two rows: A at dest +0 (x=L0, u=L1, v=L2), B at dest +2 (x=L4, u=L5, v=L6); L3 / L7 hold shared
// transient constants.
template <std::uint32_t AX, std::uint32_t AU, std::uint32_t AV, std::uint32_t BX, std::uint32_t BU, std::uint32_t BV>
inline void pair_body() {
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);  // x2
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);  // t = x2 B + A
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);  // xlog2 = x t + 127
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4);
    TTI_SFPLOADI(K1, 0, BF16_C3);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);  // u = max(u, 0) (min lands in the read-only LReg9)
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);  // u = min(u, 254)
    TTI_SFPSWAP(0, L_254, BU, 1);
    // _float_to_int32_for_exp_21f_
    TTI_SFPEXEXP(0, AU, AV, 0);
    TTI_SFPEXEXP(0, BU, BV, 0);
    TTI_SFPEXMAN(0, AU, AU, 0);
    TTI_SFPEXMAN(0, BU, BU, 0);
    TTI_SFPSHFT(0, AV, AU, 0);
    TTI_SFPSHFT(0, BV, BU, 0);
    // exponent (biased) and mantissa of z
    TTI_SFPEXEXP(0, AU, AV, 1);
    TTI_SFPEXEXP(0, BU, BV, 1);
    TTI_SFPEXMAN(0, AU, AU, 1);
    TTI_SFPEXMAN(0, BU, BU, 1);
    TTI_SFPCAST(AU, AU, 0);
    TTI_SFPCAST(BU, BU, 0);
    // 2^f polynomial (Horner), p in the x register
    TTI_SFPMAD(AU, K0, K1, AX, 0);
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);  // v = e = 2^z
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);  // d = e + 1
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);  // r ~ 1/d
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);  // eps = 1 - d r
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPLOAD(AV, 0, ADDR_MOD_7, 0);  // reload x
    TTI_SFPLOAD(BV, 0, ADDR_MOD_7, 2);
    TTI_SFPMAD(AX, AX, AX, AX, 0);  // eps + eps^2
    TTI_SFPMAD(BX, BX, BX, BX, 0);
    TTI_SFPMUL(AV, AU, L_ZERO, AV, 0);  // xr = x r
    TTI_SFPMUL(BV, BU, L_ZERO, BV, 0);
    TTI_SFPMAD(AV, AX, AV, AV, 0);  // res = xr + xr (eps + eps^2)
    TTI_SFPMAD(BV, BX, BV, BV, 0);
    TTI_SFPSTORE(AV, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(BV, 0, ADDR_MOD_7, 2);
    TTI_INCRWC(0, 4, 0, 0);
}

// V3: as V2, but the exp_21f float->int split (EXEXP, EXMAN, SHFT, EXEXP, EXMAN, CAST: 6 simple ops) is replaced by
// E = trunc(xlog2) on the round sub-unit (SFPSTOCHRND, round-to-zero, 0..255), f = xlog2 - E (exact), and the
// same degree-4 polynomial with its 2^-23k scaling removed (c_k 2^23k, exact power-of-two rescale), then
// setexp(p, E). f is exact here where exp_21f truncated it to 23 fractional bits.
constexpr std::uint16_t BF16_C2_U = 0x3E77;                  // GELU_TANH_FAST_EXP2_C2 * 2^46
constexpr std::uint16_t BF16_C3_U = 0x3D56;                  // GELU_TANH_FAST_EXP2_C3 * 2^69
constexpr std::uint16_t BF16_C4_U = 0x3C5C;                  // GELU_TANH_FAST_EXP2_C4 * 2^92
constexpr float C1_U = GELU_TANH_FAST_EXP2_C1 * 8388608.0f;  // * 2^23, L14
constexpr std::uint32_t STOCH_RND_ZERO = 2, STOCH_TO_UINT8 = 2;

template <std::uint32_t AX, std::uint32_t AU, std::uint32_t AV, std::uint32_t BX, std::uint32_t BU, std::uint32_t BV>
inline void pair_body_v3() {
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4_U);
    TTI_SFPLOADI(K1, 0, BF16_C3_U);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);
    TTI_SFPSWAP(0, L_254, BU, 1);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, AU, AV, STOCH_TO_UINT8);  // E = trunc(xlog2)
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, BU, BV, STOCH_TO_UINT8);
    TTI_SFPCAST(AV, AX, 0);
    TTI_SFPCAST(BV, BX, 0);
    TTI_SFPMAD(AX, L_ONE, AU, AU, NEG_A);  // f = xlog2 - E
    TTI_SFPMAD(BX, L_ONE, BU, BU, NEG_A);
    TTI_SFPMAD(AU, K0, K1, AX, 0);
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2_U);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPLOAD(AV, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BV, 0, ADDR_MOD_7, 2);
    TTI_SFPMAD(AX, AX, AX, AX, 0);
    TTI_SFPMAD(BX, BX, BX, BX, 0);
    TTI_SFPMUL(AV, AU, L_ZERO, AV, 0);
    TTI_SFPMUL(BV, BU, L_ZERO, BV, 0);
    TTI_SFPMAD(AV, AX, AV, AV, 0);
    TTI_SFPMAD(BV, BX, BV, BV, 0);
    TTI_SFPSTORE(AV, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(BV, 0, ADDR_MOD_7, 2);
    TTI_INCRWC(0, 4, 0, 0);
}

// V4: V3 with the tail on SFPLOADMACRO. The per-row tail "reload x; xr = x r; res = xr + xr E2; store" (4 issued
// + the E2 MAD) becomes r' = r + r E2 (regular MAD) then one SFPLOADMACRO per row that reloads x into its VD and
// schedules res = r' * VD on the MAD sub-unit (template: SFPMUL with VA = that row's r' LReg, VB <- VD) and the
// SFPSTORE of VD two cycles later. Issued per row: 2 instead of 4. Cycle-counted delays, so nothing is left
// pending at the end of a face.
constexpr std::uint32_t LM_A_VD = 2, LM_B_VD = 6;  // AV / BV

template <std::uint32_t AX, std::uint32_t AU, std::uint32_t AV, std::uint32_t BX, std::uint32_t BU, std::uint32_t BV>
inline void pair_body_v4() {
    static_assert(AV == LM_A_VD && BV == LM_B_VD);
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4_U);
    TTI_SFPLOADI(K1, 0, BF16_C3_U);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);
    TTI_SFPSWAP(0, L_254, BU, 1);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, AU, AV, STOCH_TO_UINT8);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, BU, BV, STOCH_TO_UINT8);
    TTI_SFPCAST(AV, AX, 0);
    TTI_SFPCAST(BV, BX, 0);
    TTI_SFPMAD(AX, L_ONE, AU, AU, NEG_A);
    TTI_SFPMAD(BX, L_ONE, BU, BU, NEG_A);
    TTI_SFPMAD(AU, K0, K1, AX, 0);
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2_U);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);  // eps = 1 - d r
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPMAD(AX, AX, AX, AX, 0);  // E2 = eps + eps^2
    TTI_SFPMAD(BX, BX, BX, BX, 0);
    TTI_SFPMAD(AU, AX, AU, AU, 0);  // r' = r + r E2
    TTI_SFPMAD(BU, BX, BU, BU, 0);
    // Reload x into AV / BV; macro schedules VD = r' * VD next cycle and stores VD two cycles after that.
    // The next issued instruction must not target the MAD sub-unit (a scheduled MAD wins, the regular one is
    // dropped): it is the next pair's SFPLOADI / SFPLOADs.
    TTI_SFPLOADMACRO((0 << 2) | (AV & 3), 0, ADDR_MOD_7, 0 | (AV >> 2));
    TTI_SFPLOADMACRO((1 << 2) | (BV & 3), 0, ADDR_MOD_7, 2 | (BV >> 2));
    TTI_INCRWC(0, 4, 0, 0);
}

// V5: V4 with the per-pair dest advance done by macro B's ADDR_MOD_6 (dest += 4) instead of an INCRWC.
template <std::uint32_t AX, std::uint32_t AU, std::uint32_t AV, std::uint32_t BX, std::uint32_t BU, std::uint32_t BV>
inline void pair_body_v5() {
    static_assert(AV == LM_A_VD && BV == LM_B_VD);
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4_U);
    TTI_SFPLOADI(K1, 0, BF16_C3_U);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);
    TTI_SFPSWAP(0, L_254, BU, 1);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, AU, AV, STOCH_TO_UINT8);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, BU, BV, STOCH_TO_UINT8);
    TTI_SFPCAST(AV, AX, 0);
    TTI_SFPCAST(BV, BX, 0);
    TTI_SFPMAD(AX, L_ONE, AU, AU, NEG_A);
    TTI_SFPMAD(BX, L_ONE, BU, BU, NEG_A);
    TTI_SFPMAD(AU, K0, K1, AX, 0);
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2_U);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);  // eps = 1 - d r
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPMAD(AX, AX, AX, AX, 0);  // E2 = eps + eps^2
    TTI_SFPMAD(BX, BX, BX, BX, 0);
    TTI_SFPMAD(AU, AX, AU, AU, 0);  // r' = r + r E2
    TTI_SFPMAD(BU, BX, BU, BU, 0);
    // Reload x into AV / BV; macro schedules VD = r' * VD next cycle and stores VD two cycles after that.
    // The next issued instruction must not target the MAD sub-unit (a scheduled MAD wins, the regular one is
    // dropped): it is the next pair's SFPLOADI / SFPLOADs.
    TTI_SFPLOADMACRO((0 << 2) | (AV & 3), 0, ADDR_MOD_7, 0 | (AV >> 2));
    TTI_SFPLOADMACRO((1 << 2) | (BV & 3), 0, ADDR_MOD_6, 2 | (BV >> 2));  // ADDR_MOD_6: dest += 4 (the pair)
}

// V6: V5 with the reciprocal series carried to eps^3 (error eps^4, unbiased at fp32 like the baseline's two
// Newton steps; V3-V5's eps^3 truncation leaves r' up to ~1 fp32 ulp low).
template <std::uint32_t AX, std::uint32_t AU, std::uint32_t AV, std::uint32_t BX, std::uint32_t BU, std::uint32_t BV>
inline void pair_body_v6() {
    static_assert(AV == LM_A_VD && BV == LM_B_VD);
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4_U);
    TTI_SFPLOADI(K1, 0, BF16_C3_U);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);
    TTI_SFPSWAP(0, L_254, BU, 1);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, AU, AV, STOCH_TO_UINT8);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, BU, BV, STOCH_TO_UINT8);
    TTI_SFPCAST(AV, AX, 0);
    TTI_SFPCAST(BV, BX, 0);
    TTI_SFPMAD(AX, L_ONE, AU, AU, NEG_A);
    TTI_SFPMAD(BX, L_ONE, BU, BU, NEG_A);
    TTI_SFPMAD(AU, K0, K1, AX, 0);
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2_U);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);  // eps = 1 - d r
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPMAD(AX, AX, AX, AV, 0);  // E2 = eps + eps^2 (d is dead)
    TTI_SFPMAD(BX, BX, BX, BV, 0);
    TTI_SFPMAD(AV, AX, AX, AX, 0);  // E3 = E2 eps + eps
    TTI_SFPMAD(BV, BX, BX, BX, 0);
    TTI_SFPMAD(AU, AX, AU, AU, 0);  // r' = r + r E2
    TTI_SFPMAD(BU, BX, BU, BU, 0);
    // Reload x into AV / BV; macro schedules VD = r' * VD next cycle and stores VD two cycles after that.
    // The next issued instruction must not target the MAD sub-unit (a scheduled MAD wins, the regular one is
    // dropped): it is the next pair's SFPLOADI / SFPLOADs.
    TTI_SFPLOADMACRO((0 << 2) | (AV & 3), 0, ADDR_MOD_7, 0 | (AV >> 2));
    TTI_SFPLOADMACRO((1 << 2) | (BV & 3), 0, ADDR_MOD_6, 2 | (BV >> 2));  // ADDR_MOD_6: dest += 4 (the pair)
}

inline void init_tail_macros() {
    // InstructionTemplate[0] / [1]: SFPMUL VA = r' of row A (L1) / row B (L5), VC = 0; VB, VD come from the macro.
    TTI_SFPMUL(1, 0, L_ZERO, 12, 0);
    TTI_SFPMUL(5, 0, L_ZERO, 13, 0);
    TTI_SFPNOP;
    constexpr std::uint32_t mad0 = 0x80 | (0 << 3) | (4 + 0);  // VB <- VD, delay 0, template 0
    constexpr std::uint32_t mad1 = 0x80 | (0 << 3) | (4 + 1);
    constexpr std::uint32_t store = (2 << 3) | 3;  // SFPSTORE VD, delay 2
    TTI_SFPLOADI(0, 10, (mad0 << 8));              // SFPLOADI_MOD0_LOWER
    TTI_SFPLOADI(0, 8, (store << 8));              // SFPLOADI_MOD0_UPPER (round byte 0)
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, 10, (mad1 << 8));
    TTI_SFPLOADI(0, 8, (store << 8));
    TTI_SFPCONFIG(0, 4 + 1, 0);
    // Misc: StoreMod0 0, UsesLoadMod0ForStore for macros 0/1, all delays in cycles.
    TTI_SFPCONFIG(0x030, 8, 1);
    TTI_SFPNOP;
}

}  // namespace gelu_lm

template <uint32_t V>
inline void gelu_lm_init() {
    gelu_tanh_fast_init();
    if constexpr (V >= 2) {
        TTI_SFPLOADI(0, 0, gelu_lm::BF16_254);
        TTI_SFPCONFIG(0, gelu_lm::L_254, 0);
        TTI_SFPNOP;
    }
    if constexpr (V >= 3) {
        sfpi::vConstFloatPrgm2 = gelu_lm::C1_U;
    }
    if constexpr (V >= 4) {
        gelu_lm::init_tail_macros();
    }
    if constexpr (V >= 5) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    }
}

template <uint32_t V>
inline void gelu_lm_face() {
    if constexpr (V == 1) {
        calculate_gelu_tanh_fast<true, 8>();
    } else if constexpr (V == 2) {
#pragma GCC unroll 4
        for (int pair = 0; pair < 4; ++pair) {
            gelu_lm::pair_body<0, 1, 2, 4, 5, 6>();
        }
    } else if constexpr (V == 3) {
#pragma GCC unroll 4
        for (int pair = 0; pair < 4; ++pair) {
            gelu_lm::pair_body_v3<0, 1, 2, 4, 5, 6>();
        }
    } else if constexpr (V == 4) {
#pragma GCC unroll 4
        for (int pair = 0; pair < 4; ++pair) {
            gelu_lm::pair_body_v4<0, 1, 2, 4, 5, 6>();
        }
    } else if constexpr (V == 5) {
#pragma GCC unroll 4
        for (int pair = 0; pair < 4; ++pair) {
            gelu_lm::pair_body_v5<0, 1, 2, 4, 5, 6>();
        }
    } else if constexpr (V == 6) {
#pragma GCC unroll 4
        for (int pair = 0; pair < 4; ++pair) {
            gelu_lm::pair_body_v6<0, 1, 2, 4, 5, 6>();
        }
    }
}

}  // namespace ckernel::sfpu
