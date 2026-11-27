// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu compute.
//
// Per block (op_design.md -> Block schedule):
//   untilize_rows_block   (TILE regime) compute_kernel_lib::untilize: input tiles -> row-major
//                         tile-rows (the reader copies them into the cb_rows slot).
//   windowed_tilize_block compute_kernel_lib::tilize with stick pages: per tile-row 4 windows,
//                         window p = sticks [32m + p, 32m + p + 32) = x[t - 3 + p]. The shift is a
//                         slide-by-pop on the slot (pops 1,1,1,29; 1,1,1,32 on the block's last
//                         tile-row) — exact data movement, fidelity-independent.
//   conv_silu_block       y = silu(sum_p window_p * rowbcast(tap tile, row 2p)) in DEST, one pack;
//                         the MACs run on the math thread, SiLU on the packer thread. Output tiles
//                         are acquired, packed and pushed in groups of out_group_tiles (the writer
//                         waits the same quantum): the writer stores group g while the packer runs
//                         SiLU on group g+1, and the math thread's MACs for group g+1 overlap the
//                         SiLU of group g in the other DEST half.
//
// RAW-LLK DEVIATION (conv_silu_block): eltwise_chain cannot express it.
//   (1) chain.inl:3397 — in a DEST-accumulating chain every DEST-only (SFPU) element runs on a
//       transient lane at slot >= 1, never on the sticky accumulated D0, so "accumulate 4 products,
//       then SiLU the sum" is not one chain;
//   (2) BinaryFpu::exec calls mul_tiles_bcast<bt>(...) without bcast_row_idx (chain.inl:1469), so the
//       packed tap tile (tap p in row 2p) is not addressable.
//   Two chains would need a pack + unpack of the pre-activation through an extra CB (rejected:
//   fold into the accumulator). Gap closer: a chain option to apply post-accumulation DEST-only
//   elements on the sticky slot, plus bcast_row_idx on BinaryFpu.
//   The block op below relies on ELWMUL accumulating into DEST (llk_math_eltwise_binary.h: the
//   standard ELWMUL path issues no ZEROACC; fidelity phases accumulate) and on DEST being zero at
//   acquire (cleared by the packer on release) — the same precondition every mul_tiles has.
//
// RAW-LLK DEVIATION (SiLU body): compute_kernel_lib::apply_activation_from_pack /
//   ActivationApplyHelper<SILU> dispatch only the library's fixed calculate_silu body
//   (ckernel_sfpu_silu.h: exp_21f + clamp + 3-op float->int + reciprocal with a masked Newton step,
//   26 SFPU instructions per row; one pass over conv1d-004's 96 tiles per core = 79.2 us of SFPU).
//   The helper has no slot for another SFPU body, so silu_from_pack below mirrors
//   apply_activation_from_pack (math->pack semaphore wait, packer DEST-offset flip, per-tile
//   _llk_math_eltwise_unary_sfpu_params_ sequence, SFPU-done stall) and dispatches either the library
//   body (byte-for-byte the helper's sequence) or, for a 16-bit DEST when the plan selects it
//   (Plan.silu_body), silu_pack::silu_bf16_rows — 21% less SFPU time per pass AND correctly rounded
//   for ~99.4% of inputs vs ~90%. A 32-bit DEST always runs the library body. Gap closer: an
//   apply_activation_from_pack overload taking a caller SFPU functor.

// Stage zones (MaybeDeviceZoneScope, PERMANENT — perf tournaments rank stages from them; free when the
// profiler is off). Per block (= one tile-row at the default block_rows): compute_rows_wait (unpack
// starved on the reader), compute_tilize (4 windows), compute_windows_wait, compute_conv_silu (MACs on
// math, SiLU + pack on the packer, incl. the packer's cb_output_tiles reservations); TILE regime:
// compute_untilize; once: compute_taps_wait.
// Measurement-only ablations (compiled out unless defined; wrong results by construction):
// GQKV_ABLATE_SILU, GQKV_SILU_REPEAT, GQKV_ABLATE_TILIZE (windows: CB sync only),
// GQKV_ABLATE_MAC (no products; DEST stays zero).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

#ifdef TRISC_PACK
// ---- SiLU on the packer thread, 16-bit DEST: silu(x) = x / (1 + exp(-x)) ----
namespace silu_pack {
// exp(-x) = 2^y, y = -x * log2(e). ybias = y + 639 is clamped to [512, 766]; in that binade its fp32
// encoding has exponent 9 and mantissa (y + 127) * 2^14, so shifting the encoding left by 9 drops the
// exponent and leaves (y + 127) * 2^23 — the bit pattern z = 2^floor(y) * (1 + frac(y)) (Schraudolph;
// the library exp_21f builds the same z with exexp / exman / shft). The correction 2^frac / (1 + frac)
// is a quartic in m = 1 + frac = setexp(z, 127), minimax in relative error on [1, 2): 1.1e-4, vs 1.7e-3
// for exp_21f's quadratic. Then 1 / (1 + z * g(m)) by approx_recip + one Newton step in the form
// r -= r * (d * r - 1) (d <= 2^127 is finite, so no NaN mask: approx_recip(d >= 2^126) = 0 -> r = 0),
// times x, rounded to bf16 (SFPSTORE into a 16-bit DEST truncates).
// Offline model (every finite bf16 x in [-95, 95], and normal inputs at 4 scales): correctly rounded
// for 99.3-99.5% of inputs vs 90-98% for the library body, with the bias of exact rounding; never
// more than 1 bf16 ULP off outside the x < -87 underflow tail, where both bodies return 0.
constexpr float NEG_LOG2E = -1.44269504088896341f;
constexpr float Y_BIAS = 639.0f;  // 127 (exponent bias) + 512 (pins ybias's binade: exponent 9)
constexpr float Y_LO = 512.0f;    // y + 127 >= 0: exp(-x) -> 0 for x > 88.03 (z = 0)
constexpr float Y_HI = 766.0f;    // y + 127 <= 254: z <= 2^127 stays finite
constexpr uint32_t Z_SHIFT = 9;   // log2(512): the exponent bits the shift drops
// g(m) = 2^(m - 1) / m on [1, 2), Horner g4 -> g0 (Remez, relative error 1.1e-4)
constexpr float G4 = 0.08239613473415375f;
constexpr float G3 = -0.597297728061676f;
constexpr float G2 = 1.7835139036178589f;
constexpr float G1 = -2.405177116394043f;
constexpr float G0 = 2.136453628540039f;

inline void init() {
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = NEG_LOG2E;
    sfpi::vConstFloatPrgm1 = Y_BIAS;
    sfpi::vConstFloatPrgm2 = G0;
}

// Two rows per step with the statements interleaved, so each dependent SFPMAD's result latency is
// covered by the other row's instruction (BH scoreboards dependent SFPU ops, so a one-row chain
// stalls on every MAD: the same arithmetic one row per step cost more per pass despite fewer
// instructions than the library body). Measured SFPU time of one pass over conv1d-004's 96 tiles
// per core: 62.8 us vs 79.2 us for the library body (~27.5 vs ~34.7 cycles per row); x is re-read from DEST
// for the final product, and each polynomial coefficient is loaded once per pair and shared by both
// rows (peak 8 live LRegs: za, zb, ma, mb, ga, gb and two coefficients).
template <int ITERATIONS>
inline void silu_bf16_rows() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat ya = sfpi::dst_reg[0] * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat yb = sfpi::dst_reg[1] * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        ya = sfpi::max(ya, sfpi::vFloat(Y_LO));
        yb = sfpi::max(yb, sfpi::vFloat(Y_LO));
        ya = sfpi::min(ya, sfpi::vFloat(Y_HI));
        yb = sfpi::min(yb, sfpi::vFloat(Y_HI));
        sfpi::vFloat za = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(ya) << Z_SHIFT);
        sfpi::vFloat zb = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(yb) << Z_SHIFT);
        sfpi::vFloat ma = sfpi::setexp(za, 127);
        sfpi::vFloat mb = sfpi::setexp(zb, 127);
        sfpi::vFloat ga, gb;
        {
            sfpi::vFloat g4 = G4, g3 = G3;
            ga = ma * g4 + g3;
            gb = mb * g4 + g3;
        }
        {
            sfpi::vFloat g2 = G2;
            ga = ga * ma + g2;
            gb = gb * mb + g2;
        }
        {
            sfpi::vFloat g1 = G1;
            ga = ga * ma + g1;
            gb = gb * mb + g1;
        }
        ga = ga * ma + sfpi::vConstFloatPrgm2;
        gb = gb * mb + sfpi::vConstFloatPrgm2;
        sfpi::vFloat da = za * ga + 1.0f;  // 1 + exp(-x)
        sfpi::vFloat db = zb * gb + 1.0f;
        sfpi::vFloat ra = sfpi::approx_recip(da);
        sfpi::vFloat rb = sfpi::approx_recip(db);
        sfpi::vFloat ea = da * ra - 1.0f;
        sfpi::vFloat eb = db * rb - 1.0f;
        ra = ra - ra * ea;
        rb = rb - rb * eb;
        sfpi::vFloat oa = sfpi::dst_reg[0] * ra;
        sfpi::vFloat ob = sfpi::dst_reg[1] * rb;
        oa = sfpi::convert<sfpi::vFloat16b>(oa, sfpi::RoundMode::Nearest);  // stored in DEST's own format
        ob = sfpi::convert<sfpi::vFloat16b>(ob, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = oa;
        sfpi::dst_reg[1] = ob;
        sfpi::dst_reg += 2;
    }
}
}  // namespace silu_pack

// ---- perf experiment silu_sfpu_schedule: the quartic body hand-scheduled in raw TTI ----
// RAW-LLK (experiment): bypasses the sfpi compiler's schedule of silu_pack::silu_bf16_rows (same arithmetic,
// same operand order, same rounding -> bit-identical outputs). Mechanisms exploited:
//  (1) the clamps compare against constant LRegs (SFPSWAP VC = L13 / L11 is never written back, mod 9 =
//      VD gets max), so no SFPLOADI re-materializes 512 / 766 before every swap (4 instructions per pair);
//      L11 (LCONST_neg1) is borrowed for 766 and restored after the block, -1 becomes NEGATE_VC of L10;
//  (2) SFPLOADMACRO (Blackhole multi-issue): BEG = load x + scheduled MAD y = L12 * x + L6 (MAD sub-unit),
//      FIN = load x + scheduled MUL x * r (MAD) + RND -> LReg16 (round) + STORE LReg16 (store), so the
//      first and last 4 ops of each row ride on one issue slot each; the loads re-read x from DEST.
// Pair schedule (cycles; SFPSWAP blocks the next issue slot): 38 instructions, 42 cycles per 2 rows vs
// 49 instructions (+4 swap stalls + 2 dependency stalls) for the sfpi schedule.
namespace silu_sched {
constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
constexpr uint32_t LZERO = 9, LONE = 10, LYHI = 11, LNEGLOG2E = 12, LYLO = 13, LG0 = 14;
constexpr uint32_t AM = 7;                          // ADDR_MOD_7: dest += 0 (set by the SFPU init)
constexpr uint32_t YBIAS_FP16 = 0x60FE;             // 639.0 (fp16a, exact)
constexpr uint32_t YHI_FP16 = 0x61FC;               // 766.0 (fp16a, exact)
constexpr uint32_t YLO_BF16 = 0x4400;               // 512.0
constexpr uint32_t G4_LO = 0xbf4e, G4_HI = 0x3da8;  // silu_pack::G4 .. G1, bit patterns
constexpr uint32_t G3_LO = 0xe881, G3_HI = 0xbf18;
constexpr uint32_t G2_LO = 0x4a2f, G2_HI = 0x3fe4;
constexpr uint32_t G1_LO = 0xee6c, G1_HI = 0xc019;
constexpr uint32_t MOD_NEG_VA = 1, MOD_NEG_VC = 2;
constexpr uint32_t SWAP_MIN = 1, SWAP_MAX = 9;  // VD gets min / max (VC = constant LReg: not written)

inline void init_consts() {
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = silu_pack::NEG_LOG2E;  // L12
    sfpi::vConstFloatPrgm1 = silu_pack::Y_LO;       // L13 = 512
    sfpi::vConstFloatPrgm2 = silu_pack::G0;         // L14
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATA, YHI_FP16);
    TTI_SFPCONFIG(0, LYHI, 0);  // L11 = 766 (restored to -1 by restore_neg1)
    TTI_SFPNOP;
}

inline void restore_neg1() {
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80);
    TTI_SFPCONFIG(0, LYHI, 0);
    TTI_SFPNOP;
}

inline void load_seq(uint32_t macro, uint32_t seq) {
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_LOWER, seq & 0xffff);
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_UPPER, seq >> 16);
    TTI_SFPCONFIG(0, 4 + macro, 0);
    TTI_SFPNOP;
}

// Sequence byte: [2:0] template (4..7) / 3 store, [5:3] delay (cycles after issue - 1), 0x40 VD = LReg16,
// 0x80 VB <- macro VD (MAD/round keep the template's VC).
constexpr uint32_t seq_byte(uint32_t sel, uint32_t delay, uint32_t flags) { return flags | (delay << 3) | sel; }
constexpr uint32_t SEQ_BEG = seq_byte(4, 0, 0x80) << 8;  // MAD tmpl 0 at +1
constexpr uint32_t SEQ_FIN_A =
    (seq_byte(3, 3, 0x40) << 24) | (seq_byte(7, 2, 0x40) << 16) | (seq_byte(5, 0, 0x80) << 8);  // MUL +1, RND +3, ST +4
constexpr uint32_t SEQ_FIN_B =
    (seq_byte(3, 4, 0x40) << 24) | (seq_byte(7, 3, 0x40) << 16) | (seq_byte(6, 0, 0x80) << 8);  // MUL +1, RND +4, ST +5

inline void init_macros() {
    init_consts();
    // Instruction templates via the backdoor (VD = 12 + i; DISABLE_BACKDOOR_LOAD is clear after the SFPU init).
    TTI_SFPMAD(LNEGLOG2E, 0, L6, 12, 0);  // T0: VD = L12 * VD + L6 (Y_BIAS)
    TTI_SFPMUL(L4, 0, LZERO, 13, 0);      // T1: VD = L4 (r_a) * VD
    TTI_SFPMUL(L5, 0, LZERO, 14, 0);      // T2: VD = L5 (r_b) * VD
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, LZERO, 0, 15, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // T3
    TTI_SFPNOP;
    load_seq(0, SEQ_BEG);
    load_seq(1, SEQ_FIN_A);
    load_seq(2, SEQ_FIN_B);
    // Misc: StoreMod0 0, UsesLoadMod0ForStore for macros 1 and 2, every delay counts cycles.
    TTI_SFPCONFIG(0x060, 8, 1);
    TTI_SFPNOP;
}

// Middle of a pair (rows a = dst_reg[0] in L0, b = dst_reg[1] in L1, y already computed and >= 2 cycles old):
// clamps, Schraudolph z, quartic g, d = 1 + z g, approx_recip + one Newton step; r_a -> L4, r_b -> L5.
template <bool g4_hoisted = false>
inline void pair_middle() {
    TTI_SFPSWAP(0, LYLO, L0, SWAP_MAX);  // ya = max(ya, 512)
    TTI_SFPSWAP(0, LYLO, L1, SWAP_MAX);
    TTI_SFPSWAP(0, LYHI, L0, SWAP_MIN);  // ya = min(ya, 766)
    TTI_SFPSWAP(0, LYHI, L1, SWAP_MIN);
    TTI_SFPSHFT(9, L0, L0, 5);  // za = ya << 9 (VD = VC << imm)
    TTI_SFPSHFT(9, L1, L1, 5);
    TTI_SFPSETEXP(127, L0, L2, 1);  // ma = setexp(za, 127)
    TTI_SFPSETEXP(127, L1, L3, 1);
    if constexpr (g4_hoisted) {
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, G3_HI);  // G3 low half loaded before the clamps
    } else {
        TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, G4_HI);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_USHORT, G3_LO);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, G3_HI);
    }
    TTI_SFPMAD(L2, L7, L6, L4, 0);  // ga = ma * G4 + G3
    TTI_SFPMAD(L3, L7, L6, L5, 0);
    TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_USHORT, G2_LO);
    TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, G2_HI);
    TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G2
    TTI_SFPMAD(L5, L3, L6, L5, 0);
    TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_USHORT, G1_LO);
    TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, G1_HI);
    TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G1
    TTI_SFPMAD(L5, L3, L6, L5, 0);
    TTI_SFPMAD(L4, L2, LG0, L4, 0);  // ga = ga * ma + G0
    TTI_SFPMAD(L5, L3, LG0, L5, 0);
    TTI_SFPMAD(L0, L4, LONE, L0, 0);  // da = za * ga + 1
    TTI_SFPMAD(L1, L5, LONE, L1, 0);
    TTI_SFPARECIP(0, L0, L4, sfpi::SFPARECIP_MOD1_RECIP);  // ra = approx_recip(da)
    TTI_SFPARECIP(0, L1, L5, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(L0, L4, LONE, L0, MOD_NEG_VC);  // ea = da * ra - 1
    TTI_SFPMAD(L1, L5, LONE, L1, MOD_NEG_VC);
    TTI_SFPMAD(L4, L0, L4, L4, MOD_NEG_VA);  // ra = ra - ra * ea
    TTI_SFPMAD(L5, L1, L5, L5, MOD_NEG_VA);
}

// Body 2: plain raw TTI (no macros). One face = ITERATIONS rows = ITERATIONS / 2 pairs.
template <int ITERATIONS>
inline void rows_plain() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, YBIAS_FP16);
        TTI_SFPLOAD(L0, 0, AM, 0);
        TTI_SFPLOAD(L1, 0, AM, 2);
        TTI_SFPMAD(L0, LNEGLOG2E, L6, L0, 0);  // ya = xa * NEG_LOG2E + Y_BIAS
        TTI_SFPMAD(L1, LNEGLOG2E, L6, L1, 0);
        TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, G4_LO);
        pair_middle();
        TTI_SFPLOAD(L2, 0, AM, 0);
        TTI_SFPLOAD(L3, 0, AM, 2);
        TTI_SFPMUL(L2, L4, LZERO, L2, 0);
        TTI_SFPMUL(L3, L5, LZERO, L3, 0);
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, LZERO, L2, L2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, LZERO, L3, L3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(L2, 0, AM, 0);
        TTI_SFPSTORE(L3, 0, AM, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

// Body 3: BEG / FIN load macros (init_macros).
template <int ITERATIONS>
inline void rows_macro() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, YBIAS_FP16);
        TTI_SFPLOADMACRO((0 << 2) | L0, 0, AM, 0);  // BEG a: L0 = x_a; MAD ya at +1
        TTI_SFPLOADMACRO((0 << 2) | L1, 0, AM, 2);  // BEG b
        TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, G4_LO);
        pair_middle();
        TTI_SFPLOADMACRO((1 << 2) | L2, 0, AM, 0);  // FIN a: L2 = x_a; MUL +1, RND +3, STORE +4
        TTI_SFPLOADMACRO((2 << 2) | L3, 0, AM, 2);  // FIN b: L3 = x_b; MUL +1, RND +4, STORE +5
        TTI_INCRWC(0, 4, 0, 0);
    }
    TTI_SFPNOP;  // let the last FIN's scheduled round / store drain before the face advance
    TTI_SFPNOP;
}

// Body 4: body 3 + G4 hoisted into L7 once per face (L7 is read by nothing else) + the per-pair INCRWC folded
// into FIN b's address mode (ADDR_MOD_6: dest += 4 after the load; the scheduled store keeps the address
// latched at issue). Y_BIAS moves to L4 (free until g_a; FIN a's MUL has read r_a from L4 one cycle before),
// so G3's low half can load into L6 right behind the BEGs and fill the gap the first clamp needs.
// 38 instructions, 39 cycles per pair (+2 LOADIs and 2 NOPs per face).
constexpr uint32_t AM_INC4 = 6;
constexpr uint32_t SEQ_BEG4 = seq_byte(4, 0, 0x80) << 8;

inline void init_macros4() {
    init_consts();
    TTI_SFPMAD(LNEGLOG2E, 0, L4, 12, 0);  // T0: VD = L12 * VD + L4 (Y_BIAS)
    TTI_SFPMUL(L4, 0, LZERO, 13, 0);      // T1: VD = L4 (r_a) * VD
    TTI_SFPMUL(L5, 0, LZERO, 14, 0);      // T2: VD = L5 (r_b) * VD
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, LZERO, 0, 15, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // T3
    TTI_SFPNOP;
    load_seq(0, SEQ_BEG4);
    load_seq(1, SEQ_FIN_A);
    load_seq(2, SEQ_FIN_B);
    TTI_SFPCONFIG(0x060, 8, 1);
    TTI_SFPNOP;
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

template <int ITERATIONS>
inline void rows_macro4() {
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, G4_LO);
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, G4_HI);
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_FLOATA, YBIAS_FP16);
        TTI_SFPLOADMACRO((0 << 2) | L0, 0, AM, 0);  // BEG a: L0 = x_a; MAD ya = L12 * x_a + L4 at +1
        TTI_SFPLOADMACRO((0 << 2) | L1, 0, AM, 2);  // BEG b
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_USHORT, G3_LO);
        pair_middle<true>();
        TTI_SFPLOADMACRO((1 << 2) | L2, 0, AM, 0);       // FIN a: MUL +1, RND +3, STORE +4
        TTI_SFPLOADMACRO((2 << 2) | L3, 0, AM_INC4, 2);  // FIN b: MUL +1, RND +4, STORE +5; dest += 4
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Body 5: body 4 with the quartic refit so that G3, G2, G1 are fp16 (one SFPLOADI FLOATA each instead of an
// fp32 USHORT + UPPER pair); G4 (L7, once per face) and G0 (L14) stay fp32. Minimax relative error 1.140e-4
// vs 1.112e-4 for the fp32 quartic (fit_fp16_coeffs.py). NOT bit-identical to body 1: same accuracy class.
// 35 instructions, 36 cycles per pair.
constexpr float G0_F16FIT = 2.1355555057525635f;
constexpr uint32_t G4F_LO = 0x29c0, G4F_HI = 0x3da8;  // 0.08211088180541992
constexpr uint32_t G3_FP16 = 0xb8c4;                  // -0.595703125
constexpr uint32_t G2_FP16 = 0x3f1f;                  // 1.7802734375
constexpr uint32_t G1_FP16 = 0xc0ce;                  // -2.40234375

inline void init_macros5() {
    init_macros4();
    sfpi::vConstFloatPrgm2 = G0_F16FIT;  // L14
}

inline void load_g4_f16fit() {
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, G4F_LO);
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, G4F_HI);
}

// face_overhead = true (body 5): G4 into L7 and two drain NOPs on every face; false (body 6): the caller loads
// G4 once per DEST group (L7 is read by nothing else) and drains once after the group's last tile.
template <int ITERATIONS, bool face_overhead = true>
inline void rows_macro5() {
    if constexpr (face_overhead) {
        load_g4_f16fit();
    }
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_FLOATA, YBIAS_FP16);
        TTI_SFPLOADMACRO((0 << 2) | L0, 0, AM, 0);              // BEG a: L0 = x_a; MAD ya = L12 * x_a + L4 at +1
        TTI_SFPLOADMACRO((0 << 2) | L1, 0, AM, 2);              // BEG b
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G3_FP16);  // fills the gap the first clamp needs
        TTI_SFPSWAP(0, LYLO, L0, SWAP_MAX);
        TTI_SFPSWAP(0, LYLO, L1, SWAP_MAX);
        TTI_SFPSWAP(0, LYHI, L0, SWAP_MIN);
        TTI_SFPSWAP(0, LYHI, L1, SWAP_MIN);
        TTI_SFPSHFT(9, L0, L0, 5);
        TTI_SFPSHFT(9, L1, L1, 5);
        TTI_SFPSETEXP(127, L0, L2, 1);
        TTI_SFPSETEXP(127, L1, L3, 1);
        TTI_SFPMAD(L2, L7, L6, L4, 0);  // ga = ma * G4 + G3
        TTI_SFPMAD(L3, L7, L6, L5, 0);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G2_FP16);
        TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G2
        TTI_SFPMAD(L5, L3, L6, L5, 0);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G1_FP16);
        TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G1
        TTI_SFPMAD(L5, L3, L6, L5, 0);
        TTI_SFPMAD(L4, L2, LG0, L4, 0);  // ga = ga * ma + G0
        TTI_SFPMAD(L5, L3, LG0, L5, 0);
        TTI_SFPMAD(L0, L4, LONE, L0, 0);  // da = za * ga + 1
        TTI_SFPMAD(L1, L5, LONE, L1, 0);
        TTI_SFPARECIP(0, L0, L4, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPARECIP(0, L1, L5, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(L0, L4, LONE, L0, MOD_NEG_VC);  // ea = da * ra - 1
        TTI_SFPMAD(L1, L5, LONE, L1, MOD_NEG_VC);
        TTI_SFPMAD(L4, L0, L4, L4, MOD_NEG_VA);  // ra = ra - ra * ea
        TTI_SFPMAD(L5, L1, L5, L5, MOD_NEG_VA);
        TTI_SFPLOADMACRO((1 << 2) | L2, 0, AM, 0);       // FIN a
        TTI_SFPLOADMACRO((2 << 2) | L3, 0, AM_INC4, 2);  // FIN b; dest += 4
    }
    if constexpr (face_overhead) {
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

template <int ITERATIONS>
inline void rows_macro6() {
    rows_macro5<ITERATIONS, false>();
}

// Body 7 (PRECISION TRADE, menu only): body 6's schedule with a CUBIC correction g(m) = G3 m^3 + G2 m^2 + G1 m + G0
// (minimax relative error 6.67e-4 vs 1.14e-4; G2, G1 fp16, G3 in L7 once per group, G0 in L14).
// 30 instructions, 33 cycles per pair.
constexpr float G0_CUBIC = 1.7750027179718018f;
constexpr uint32_t C3_LO = 0x921d, C3_HI = 0xbdd9;  // -0.10623572021722794
constexpr uint32_t C2_FP16 = 0x39a7;                // 0.70654296875
constexpr uint32_t C1_FP16 = 0xbd81;                // -1.3759765625

inline void init_macros7() {
    init_macros4();
    sfpi::vConstFloatPrgm2 = G0_CUBIC;  // L14
}

inline void load_g3_cubic() {
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, C3_LO);
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, C3_HI);
}

template <int ITERATIONS>
inline void rows_macro7() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_FLOATA, YBIAS_FP16);
        TTI_SFPLOADMACRO((0 << 2) | L0, 0, AM, 0);
        TTI_SFPLOADMACRO((0 << 2) | L1, 0, AM, 2);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, C2_FP16);
        TTI_SFPSWAP(0, LYLO, L0, SWAP_MAX);
        TTI_SFPSWAP(0, LYLO, L1, SWAP_MAX);
        TTI_SFPSWAP(0, LYHI, L0, SWAP_MIN);
        TTI_SFPSWAP(0, LYHI, L1, SWAP_MIN);
        TTI_SFPSHFT(9, L0, L0, 5);
        TTI_SFPSHFT(9, L1, L1, 5);
        TTI_SFPSETEXP(127, L0, L2, 1);
        TTI_SFPSETEXP(127, L1, L3, 1);
        TTI_SFPMAD(L2, L7, L6, L4, 0);  // ga = ma * G3 + G2
        TTI_SFPMAD(L3, L7, L6, L5, 0);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, C1_FP16);
        TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G1
        TTI_SFPMAD(L5, L3, L6, L5, 0);
        TTI_SFPMAD(L4, L2, LG0, L4, 0);  // ga = ga * ma + G0
        TTI_SFPMAD(L5, L3, LG0, L5, 0);
        TTI_SFPMAD(L0, L4, LONE, L0, 0);
        TTI_SFPMAD(L1, L5, LONE, L1, 0);
        TTI_SFPARECIP(0, L0, L4, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPARECIP(0, L1, L5, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(L0, L4, LONE, L0, MOD_NEG_VC);
        TTI_SFPMAD(L1, L5, LONE, L1, MOD_NEG_VC);
        TTI_SFPMAD(L4, L0, L4, L4, MOD_NEG_VA);
        TTI_SFPMAD(L5, L1, L5, L5, MOD_NEG_VA);
        TTI_SFPLOADMACRO((1 << 2) | L2, 0, AM, 0);
        TTI_SFPLOADMACRO((2 << 2) | L3, 0, AM_INC4, 2);
    }
}
}  // namespace silu_sched
#endif

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_window_tiles = 2;
constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t WINDOW_STICKS = TILE_DIM + HALO_ROWS;  // every window of a tile-row lies in these
constexpr uint32_t TAP_ROW_STRIDE = 2;                    // tap p lives in row 2p of the packed tap tile

// SiLU body selector (compute CT arg 4, Plan.silu_body in the program descriptor): 0 = the library
// calculate_silu, 1 = silu_pack::silu_bf16_rows. A 32-bit DEST always runs the library body.
constexpr uint32_t SILU_LIBRARY = 0;

// perf experiment: the body comes from GQKV_SILU_BODY when defined (0 library, 1 sfpi quartic, 2 raw-TTI
// quartic, 3 raw-TTI quartic + load macros), else from the plan's CT arg (0 library, 1 quartic).
// A 32-bit DEST always runs the library body.
template <uint32_t silu_variant>
constexpr uint32_t silu_body() {
    if constexpr (DST_ACCUM_MODE) {
        return 0;
    } else {
#ifdef GQKV_SILU_BODY
        return GQKV_SILU_BODY;
#else
        return silu_variant;
#endif
    }
}

template <uint32_t silu_variant>
constexpr bool use_bf16_silu() {
    return silu_body<silu_variant>() != SILU_LIBRARY;
}

template <uint32_t silu_variant>
ALWI void silu_init() {
    constexpr uint32_t body = silu_body<silu_variant>();
    if constexpr (body == 1) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_pack::init)));
    } else if constexpr (body == 2) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_sched::init_consts)));
    } else if constexpr (body == 3) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_sched::init_macros)));
    } else if constexpr (body == 4) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_sched::init_macros4)));
    } else if constexpr (body == 5 || body == 6) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_sched::init_macros5)));
    } else if constexpr (body == 7) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_sched::init_macros7)));
    } else {
        compute_kernel_lib::ActivationInitHelper<KernelActivation::SILU>::init();
    }
}

// The raw bodies borrow L11 (LCONST_neg1) for 766: put -1 back once the block's SiLU work is done.
template <uint32_t silu_variant>
ALWI void silu_uninit() {
    constexpr uint32_t body = silu_body<silu_variant>();
    if constexpr (body >= 2) {
        PACK((silu_sched::restore_neg1()));
    }
}

#ifdef TRISC_PACK
// SiLU on one DEST tile from the packer thread: _llk_math_eltwise_unary_sfpu_params_(body, tile, RC)
// with its per-tile STALLWAIT(STALL_SFPU, MATH) optional (see silu_from_pack).
template <uint32_t body, bool tile_stall>
inline void silu_tile_from_pack(uint32_t tile) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);
    if constexpr (tile_stall) {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    }
    if constexpr (body == 1) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_pack::silu_bf16_rows<8>, VectorMode::RC);
    } else if constexpr (body == 2) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_plain<8>, VectorMode::RC);
    } else if constexpr (body == 3) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_macro<8>, VectorMode::RC);
    } else if constexpr (body == 4) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_macro4<8>, VectorMode::RC);
    } else if constexpr (body == 5) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_macro5<8>, VectorMode::RC);
    } else if constexpr (body == 6) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_macro6<8>, VectorMode::RC);
    } else if constexpr (body == 7) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_sched::rows_macro7<8>, VectorMode::RC);
    } else {
        _llk_math_eltwise_sfpu_apply_vector_mode_(ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, 8>, VectorMode::RC);
    }
    ckernel::math::clear_dst_reg_addr();
}
#endif

// apply_activation_from_pack<SILU> with a selectable body and per-tile stall (see RAW-LLK DEVIATION
// above): replaces tile_regs_wait() — math->pack semaphore wait, packer DEST-offset flip, SiLU on
// each of `tiles` DEST tiles, SFPU-done stall before pack.
// tile_stall: the library start (_llk_math_eltwise_sfpu_start_) stalls the SFPU until the FPU pipe
// is idle before EVERY tile. From the packer that wait is redundant — math posts MATH_PACK only after
// STALLWAIT(MATH | WAIT_SFPU) (set_math_semaphores), so the whole DEST half is written once the
// SEMWAIT returns — and it is costly: the FPU is then busy with the NEXT group's MACs in the other
// DEST half, so each tile's SiLU waits for a gap in that stream while math's own commit waits for
// the SFPU (SILU_TILE_STALL in the program descriptor).
template <uint32_t silu_variant, bool tile_stall>
ALWI void silu_from_pack(uint32_t tiles) {
#ifdef GQKV_ABLATE_SILU
    // Perf ablation only (test_perf_r3.py; wrong results): the handshake without any SFPU work.
    tile_regs_wait();
    return;
#endif
#ifdef GQKV_SILU_REPEAT
    // Perf measurement only (test_perf_r3.py; wrong results): the body runs GQKV_SILU_REPEAT times
    // per tile, so the SFPU cost of one pass is a device-time difference.
    constexpr uint32_t passes = GQKV_SILU_REPEAT;
#else
    constexpr uint32_t passes = 1;
#endif
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    constexpr uint32_t body = silu_body<silu_variant>();
    if constexpr (body == 6) {
        PACK((silu_sched::load_g4_f16fit()));
    } else if constexpr (body == 7) {
        PACK((silu_sched::load_g3_cubic()));
    }
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t i = 0; i < tiles; ++i) {
            PACK((silu_tile_from_pack<body, tile_stall>(i)));
        }
    }
    if constexpr (body == 6 || body == 7) {
        PACK(TTI_SFPNOP);  // drain the last FIN's scheduled round / store
        PACK(TTI_SFPNOP);
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

// One window: tilize 32 sticks from the slot's read pointer, then pop `pages` sticks.
template <uint32_t block_ch_tiles, InitUninitMode init_mode, ReconfigureRegisterDatatypeMode reconfig>
ALWI void tilize_window(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t pages) {
#ifdef GQKV_ABLATE_TILIZE
    // Perf ablation only: the helper's CB sequence (reserve, push, pop) without the tilize.
    windows.reserve_back(block_ch_tiles);
    windows.push_back(block_ch_tiles);
    rows.pop_front(pages);
#else
    compute_kernel_lib::tilize<block_ch_tiles, cb_rows, cb_window_tiles, init_mode, WaitMode::NoWait, reconfig>(
        rows, windows, 1, pages);
#endif
}

// windowed_tilize_block: rows_this_block tile-rows x 4 windows x block_ch_tiles tiles, layout
// [m][p][c] in cb_window_tiles. One tilize init / uninit per block; one wait per tile-row.
template <uint32_t block_ch_tiles>
void windowed_tilize_block(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t rows_this_block) {
    constexpr auto Keep = ReconfigureRegisterDatatypeMode::NoReconfigure;
    for (uint32_t m = 0; m < rows_this_block; ++m) {
        {
            MaybeDeviceZoneScope("compute_rows_wait");
            rows.wait_front(WINDOW_STICKS);
        }
        MaybeDeviceZoneScope("compute_tilize");
        // p = 0
        if (m == 0) {
            tilize_window<
                block_ch_tiles,
                InitUninitMode::InitOnly,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(rows, windows, 1);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 1, 2
        for (uint32_t p = 1; p + 1 < TAP_COUNT; ++p) {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 3: release the rest of the tile-row; the block's last tile-row also releases the
        // 3 trailing sticks, so the block pops exactly its slot (32 * rows + 3).
        if (m + 1 == rows_this_block) {
            tilize_window<block_ch_tiles, InitUninitMode::UninitOnly, Keep>(rows, windows, TILE_DIM);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, TILE_DIM - HALO_ROWS);
        }
    }
}

// conv_silu_block: rows_this_block x block_ch_tiles output tiles. Output tile (m, c) =
// silu(sum_{p=0..3} window[m*4w + p*w + c] * rowbcast(tap_tile[c], row 2p)), the 4 products
// accumulated in one DEST slot in fixed order p = 0..3, SiLU applied in DEST, packed once.
// Each tile-row's tiles go out in groups of out_group_tiles (the last group of a row may be short);
// a row's groups sum to block_ch_tiles, so no reservation straddles the end of cb_output_tiles.
template <uint32_t block_ch_tiles, uint32_t out_group_tiles, uint32_t silu_variant, bool silu_tile_stall>
void conv_silu_block(DataflowBuffer& windows, DataflowBuffer& outputs, uint32_t rows_this_block) {
    static_assert(
        out_group_tiles >= 1 && out_group_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT,
        "an output group is one DEST acquire");
    constexpr uint32_t dest_group = out_group_tiles;
    const uint32_t window_tiles = TAP_COUNT * block_ch_tiles * rows_this_block;
    {
        MaybeDeviceZoneScope("compute_windows_wait");
        windows.wait_front(window_tiles);
    }
    MaybeDeviceZoneScope("compute_conv_silu");

    reconfig_data_format(cb_window_tiles, cb_tap_tiles);
    pack_reconfig_data_format(cb_output_tiles);
    mul_bcast_rows_init(cb_window_tiles, cb_tap_tiles);
    // SiLU runs on the packer thread (TRISC2), overlapping the math thread's next group of MACs
    // (op_design.md Perf lamps: measured 197.7 -> 185.9 us on conv1d-004 vs SiLU on the math thread).
    silu_init<silu_variant>();

    for (uint32_t m = 0; m < rows_this_block; ++m) {
        const uint32_t row_base = m * TAP_COUNT * block_ch_tiles;
        for (uint32_t c0 = 0; c0 < block_ch_tiles; c0 += dest_group) {
            const uint32_t group = (block_ch_tiles - c0 < dest_group) ? (block_ch_tiles - c0) : dest_group;
            outputs.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                const uint32_t c = c0 + i;
#ifndef GQKV_ABLATE_MAC
                for (uint32_t p = 0; p < TAP_COUNT; ++p) {
                    mul_tiles_bcast_rows(
                        cb_window_tiles, cb_tap_tiles, row_base + p * block_ch_tiles + c, c, i, TAP_ROW_STRIDE * p);
                }
#endif
            }
            tile_regs_commit();
            // Replaces tile_regs_wait(): math->pack handshake, SiLU on each tile in DEST from the
            // packer thread, SFPU-done stall before pack.
            silu_from_pack<silu_variant, silu_tile_stall>(group);
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, cb_output_tiles);
            }
            tile_regs_release();
            outputs.push_back(group);
        }
    }
    silu_uninit<silu_variant>();
    windows.pop_front(window_tiles);
}

// untilize_rows_block (TILE regime): (rows + extra) tile-rows of input tiles -> row-major.
template <uint32_t block_ch_tiles>
void untilize_rows_block(uint32_t tile_rows) {
    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer untilized_rows(cb_untilized_rows);
    MaybeDeviceZoneScope("compute_untilize");
    compute_kernel_lib::untilize<block_ch_tiles, cb_input_tiles, cb_untilized_rows>(
        input_tiles, untilized_rows, tile_rows);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(3);  // output tiles per push
    constexpr uint32_t silu_variant = get_compile_time_arg_val(4);     // SiLU body (silu_from_pack)
#ifdef GQKV_TILE_STALL_OFF
    constexpr bool silu_tile_stall = false;  // perf experiment: per-tile STALLWAIT(STALL_SFPU, MATH) dropped
#else
    constexpr bool silu_tile_stall = get_compile_time_arg_val(5) == 1;  // per-tile SFPU-after-FPU stall
#endif

    const uint32_t core_seq_start = get_arg_val<uint32_t>(0);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    if constexpr (tile_input) {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_untilized_rows);
    } else {
        compute_kernel_hw_startup(cb_rows, cb_rows, cb_window_tiles);
    }

    DataflowBuffer rows(cb_rows);
    DataflowBuffer taps(cb_tap_tiles);
    DataflowBuffer windows(cb_window_tiles);
    DataflowBuffer outputs(cb_output_tiles);

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
        if constexpr (tile_input) {
            // A core whose range starts at t0 > 0 untilizes one extra tile-row on its first block:
            // its rows 29..31 are that block's halo.
            const uint32_t extra_rows = (block_idx == 0 && core_seq_start > 0) ? 1 : 0;
            untilize_rows_block<block_ch_tiles>(rows_this_block + extra_rows);
        }
        windowed_tilize_block<block_ch_tiles>(rows, windows, rows_this_block);
        if (block_idx == 0) {
            MaybeDeviceZoneScope("compute_taps_wait");
            taps.wait_front(block_ch_tiles);  // first use: the tap fetch hid behind the first tilize
        }
        conv_silu_block<block_ch_tiles, out_group_tiles, silu_variant, silu_tile_stall>(
            windows, outputs, rows_this_block);
    }
    taps.pop_front(block_ch_tiles);
}
