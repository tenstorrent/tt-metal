// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for multigammaln_lanczos at order p = 4.
//
// Faithful translation of the Lanczos 6-term polynomial recipe into a single
// fused TTNN compute kernel. The kernel deliberately does NOT use the SFPU
// `lgamma_tile` / `lgamma_stirling_float_tile` / `lgamma_adjusted_tile`
// helpers — the Lanczos polynomial is implemented from primitive SFPU ops.
//
// Per input tile:
//   1. Hold cb_input_tiles (cb_wait_front, NOT pop). The same input tile feeds
//      all four Lanczos sub-phases.
//   2. For each offset in {0.0, 0.5, 1.0, 1.5}, run the Lanczos polynomial
//      recipe on (a - offset) and pack the result into the matching
//      intermediate CB (cb_lgamma_a, _half, _one, _three_halves).
//   3. Pop cb_input_tiles after all four Lanczos sub-phases have read it.
//   4. Sum the four Lanczos terms into D0, add the compile-time constant
//      3 * log(pi) via add_unary_tile, and pack to cb_output_tiles.
//
// DEST budget: with fp32_dest_acc_en + half-sync, DEST holds 4 tiles per
// acquire/release. The per-offset Lanczos recipe uses D0..D3; the 4-way sum
// at the end also uses D0..D3.
//
// Algorithm — Lanczos 6-term polynomial:
//
//   For a single argument y = a - offset:
//     input  = y - 1                                          (= a - (offset + 1))
//     series = 1 + sum_{j=1..6} coef[j] / (input + j)
//     t      = input + 5.5                                    (= a - (offset - 4.5))
//
//   L(y) = (input + 0.5) * log(t) + log(series) + log(sqrt(2*pi)) - t
//
//   Algebraic identity used here (avoids needing un-logged t in DEST after
//   log_tile overwrites it):
//     L(y) = (input + 0.5) * log(t) + log(series) - input - 4.581061468643f
//   since (0.918938531357171 - 5.5) == -4.581061468643.
//
//   Zero-clamp at integer poles y == 1, y == 2 (where the polynomial has 1/0
//   blowups but the true lgamma value is 0). Implemented as
//     mask = (a != offset + 1) * (a != offset + 2)
//     result *= mask
//   (a `where`-style elementwise select — no input-value branching).
//
// Precision note: every fp32 CB used by this kernel — cb_input_tiles,
// cb_output_tiles, and the four cb_lgamma_* intermediates — is configured
// with `UnpackToDestMode::UnpackToDestFp32` in the program descriptor. This
// is required: without it, `copy_tile(cb_lgamma_*, 0, slot)` in sub-phase B
// routes through SrcA/SrcB and may TF32-truncate the mantissa.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/compute_kernel_api.h"

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_lgamma_a = 24;               // L(a - 0.0)
constexpr uint32_t cb_lgamma_a_half = 25;          // L(a - 0.5)
constexpr uint32_t cb_lgamma_a_one = 26;           // L(a - 1.0)
constexpr uint32_t cb_lgamma_a_three_halves = 27;  // L(a - 1.5)

// constexpr float -> uint32_t bit-pattern helper. Used for every fp32 scalar
// passed to a *_unary_tile() / unary_eq_tile() call (those APIs take a
// uint32_t bit-cast of the fp32 value — see binop_with_scalar.h:24).
constexpr uint32_t fp32_bits(float v) { return __builtin_bit_cast(uint32_t, v); }

// 3 * log(pi) ≈ 3.434189657547 — the (p*(p-1)/4)*log(pi) constant for p = 4.
constexpr uint32_t THREE_LOG_PI_BITS = fp32_bits(3.434189657547f);

// Lanczos coefficients (6-term).
constexpr uint32_t COEF_BITS_1 = fp32_bits(76.18009172947146f);
constexpr uint32_t COEF_BITS_2 = fp32_bits(-86.50532032941677f);
constexpr uint32_t COEF_BITS_3 = fp32_bits(24.01409824083091f);
constexpr uint32_t COEF_BITS_4 = fp32_bits(-1.231739572450155f);
constexpr uint32_t COEF_BITS_5 = fp32_bits(0.1208650973866179e-2f);
constexpr uint32_t COEF_BITS_6 = fp32_bits(-0.5395239384953e-5f);

// Algebraic simplification constant: 5.5 - 0.918938531357171 = 4.581061468643.
constexpr uint32_t LOG_TWO_PI_HALF_MINUS_5_5_BITS = fp32_bits(4.581061468643f);

constexpr uint32_t ONE_F_BITS = fp32_bits(1.0f);

// Per-offset constant table. Indexed by OFFSET_ID in {0, 1, 2, 3} corresponding
// to OFFSET in {0.0f, 0.5f, 1.0f, 1.5f}. All values are fp32 bit patterns
// (uint32_t) ready to feed to the binop_with_scalar / unary_eq APIs.
template <int OFFSET_ID>
struct LanczosOffsetConstants {
    static constexpr float OFFSET = (OFFSET_ID == 0) ? 0.0f : (OFFSET_ID == 1) ? 0.5f : (OFFSET_ID == 2) ? 1.0f : 1.5f;

    static constexpr uint32_t SUB_FOR_T_BITS = fp32_bits(OFFSET - 4.5f);           // D1 = a - (OFFSET - 4.5) = t
    static constexpr uint32_t SUB_FOR_INPUT_BITS = fp32_bits(OFFSET + 1.0f);       // D0 = a - (OFFSET + 1) = input
    static constexpr uint32_t SUB_FOR_INPUT_HALF_BITS = fp32_bits(OFFSET + 0.5f);  // D3 = a - (OFFSET + 0.5)
    static constexpr uint32_t POLE_AT_ONE_BITS = fp32_bits(OFFSET + 1.0f);         // a == OFFSET + 1  <->  y == 1
    static constexpr uint32_t POLE_AT_TWO_BITS = fp32_bits(OFFSET + 2.0f);         // a == OFFSET + 2  <->  y == 2

    // Per-j sub bits: (input + j) = a - (OFFSET + 1 - j) -> we sub `OFFSET + 1 - j`.
    // Equivalently: sub `OFFSET - (j - 1)` from `a`.
    static constexpr uint32_t SUB_J1_BITS = fp32_bits(OFFSET - 0.0f);  // j = 1
    static constexpr uint32_t SUB_J2_BITS = fp32_bits(OFFSET - 1.0f);  // j = 2
    static constexpr uint32_t SUB_J3_BITS = fp32_bits(OFFSET - 2.0f);  // j = 3
    static constexpr uint32_t SUB_J4_BITS = fp32_bits(OFFSET - 3.0f);  // j = 4
    static constexpr uint32_t SUB_J5_BITS = fp32_bits(OFFSET - 4.0f);  // j = 5
    static constexpr uint32_t SUB_J6_BITS = fp32_bits(OFFSET - 5.0f);  // j = 6
};

// ---- Per-offset Lanczos sub-procedure --------------------------------------
//
// Computes one L(a - OFFSET) and packs the result into `cb_out`. OFFSET_ID
// selects the OFFSET via the constant table above.
//
// Pre:  cb_input_tiles holds 1 tile at the front. Tile is NOT popped here —
//       caller pops once after running all four offsets.
// Post: cb_out has 1 tile pushed.
//
// DEST slot allocation across this acquire/release block:
//   D0 -> input = a - (OFFSET + 1)              then later: pole-zero mask
//   D1 -> t = a - (OFFSET - 4.5)                then: log(t)
//   D2 -> series accumulator (init 1.0)         then: log(series)
//   D3 -> per-j scratch, then `input + 0.5`, then the running L-result
//         (final D3 holds L(y) — packed at end)
template <int OFFSET_ID, uint32_t cb_out>
ALWI void lanczos_with_offset() {
    using K = LanczosOffsetConstants<OFFSET_ID>;

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();

    // ---- D1 = t = a - (OFFSET - 4.5) ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 1);
    binop_with_scalar_tile_init();
    sub_unary_tile(1, K::SUB_FOR_T_BITS);

    // ---- D1 = log(t) ----
    log_tile_init();
    log_tile(1);

    // ---- D2 = 1.0  (series accumulator) ----
    fill_tile_init();
    fill_tile(2, 1.0f);

    // ---- D2 = series = 1 + sum_{j=1..6} coef[j] / (input + j)  ----
    //   Each iteration: D3 = a - (OFFSET + 1 - j), recip, mul coef[j], add to D2.

#define MULTIGAMMALN_LANCZOS_TERM(sub_bits, coef_bits) \
    do {                                               \
        copy_tile_to_dst_init_short(cb_input_tiles);   \
        copy_tile(cb_input_tiles, 0, 3);               \
        binop_with_scalar_tile_init();                 \
        sub_unary_tile(3, (sub_bits));                 \
        recip_tile_init();                             \
        recip_tile(3);                                 \
        binop_with_scalar_tile_init();                 \
        mul_unary_tile(3, (coef_bits));                \
        add_binary_tile_init();                        \
        add_binary_tile(2, 3, 2);                      \
    } while (0)

    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J1_BITS, COEF_BITS_1);
    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J2_BITS, COEF_BITS_2);
    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J3_BITS, COEF_BITS_3);
    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J4_BITS, COEF_BITS_4);
    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J5_BITS, COEF_BITS_5);
    MULTIGAMMALN_LANCZOS_TERM(K::SUB_J6_BITS, COEF_BITS_6);

#undef MULTIGAMMALN_LANCZOS_TERM

    // ---- D2 = log(series) ----
    log_tile_init();
    log_tile(2);

    // ---- D0 = input = a - (OFFSET + 1) ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 0);
    binop_with_scalar_tile_init();
    sub_unary_tile(0, K::SUB_FOR_INPUT_BITS);

    // ---- D3 = input + 0.5 = a - (OFFSET + 0.5) ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 3);
    binop_with_scalar_tile_init();
    sub_unary_tile(3, K::SUB_FOR_INPUT_HALF_BITS);

    // ---- D3 = (input + 0.5) * log(t) ----
    mul_binary_tile_init();
    mul_binary_tile(3, 1, 3);

    // ---- D3 += log(series) ----
    add_binary_tile_init();
    add_binary_tile(3, 2, 3);

    // ---- D3 -= input ----
    sub_binary_tile_init();
    sub_binary_tile(3, 0, 3);

    // ---- D3 -= 4.581061468643  (5.5 - log(sqrt(2*pi))) ----
    binop_with_scalar_tile_init();
    sub_unary_tile(3, LOG_TWO_PI_HALF_MINUS_5_5_BITS);
    // D3 now holds L(y), BEFORE pole zeroing.

    // ---- Pole zero at y == 1  i.e.  a == OFFSET + 1 ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 0);  // D0 = a (reload)
    unary_eq_tile_init();
    unary_eq_tile(0, K::POLE_AT_ONE_BITS);  // D0 = (a == OFFSET + 1) ? 1 : 0
    binop_with_scalar_tile_init();
    rsub_unary_tile(0, ONE_F_BITS);  // D0 = 1 - D0 = (a != OFFSET + 1)
    mul_binary_tile_init();
    mul_binary_tile(3, 0, 3);  // D3 *= mask

    // ---- Pole zero at y == 2  i.e.  a == OFFSET + 2 ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 0);  // D0 = a (reload)
    unary_eq_tile_init();
    unary_eq_tile(0, K::POLE_AT_TWO_BITS);  // D0 = (a == OFFSET + 2) ? 1 : 0
    binop_with_scalar_tile_init();
    rsub_unary_tile(0, ONE_F_BITS);  // D0 = 1 - D0 = (a != OFFSET + 2)
    mul_binary_tile_init();
    mul_binary_tile(3, 0, 3);  // D3 *= mask

    // ---- Pack the per-offset L result ----
    tile_regs_commit();
    tile_regs_wait();

    pack_tile(3, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
}

// ---- Sub-phase B: 4-way sum + 3*log(pi), pack to cb_output_tiles ----------
//
// Raw API rather than `sfpu_chain + sfpu_pipeline`: the chain framework
// default-constructs each op from its type list, which discards scalar
// member values (e.g. `AddScalar<Dst::D0>{K}` loses K and adds 0). The same
// issue is documented for the Stirling-flavour kernel
// (multigammaln/kernels/multigammaln_compute.cpp:200-206). Until the helper
// forwards op instances, the final +constant must be expressed by hand.
ALWI void sum_and_add_const() {
    cb_wait_front(cb_lgamma_a, 1);
    cb_wait_front(cb_lgamma_a_half, 1);
    cb_wait_front(cb_lgamma_a_one, 1);
    cb_wait_front(cb_lgamma_a_three_halves, 1);

    cb_reserve_back(cb_output_tiles, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_lgamma_a);
    copy_tile(cb_lgamma_a, 0, 0);
    copy_tile_to_dst_init_short(cb_lgamma_a_half);
    copy_tile(cb_lgamma_a_half, 0, 1);
    copy_tile_to_dst_init_short(cb_lgamma_a_one);
    copy_tile(cb_lgamma_a_one, 0, 2);
    copy_tile_to_dst_init_short(cb_lgamma_a_three_halves);
    copy_tile(cb_lgamma_a_three_halves, 0, 3);

    // D0 = D0 + D1 + D2 + D3
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);

    // D0 += 3 * log(pi)
    binop_with_scalar_tile_init();
    add_unary_tile(0, THREE_LOG_PI_BITS);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, cb_output_tiles);

    cb_pop_front(cb_lgamma_a, 1);
    cb_pop_front(cb_lgamma_a_half, 1);
    cb_pop_front(cb_lgamma_a_one, 1);
    cb_pop_front(cb_lgamma_a_three_halves, 1);
    cb_push_back(cb_output_tiles, 1);

    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_input_tiles, cb_output_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_input_tiles, 1);

        // Sub-phase A x4 — Lanczos polynomial on (a - OFFSET) for each offset.
        // The input tile is held across all four sub-phases (NOT popped here).
        // OFFSET_ID: 0=0.0f, 1=0.5f, 2=1.0f, 3=1.5f.
        lanczos_with_offset<0, cb_lgamma_a>();
        lanczos_with_offset<1, cb_lgamma_a_half>();
        lanczos_with_offset<2, cb_lgamma_a_one>();
        lanczos_with_offset<3, cb_lgamma_a_three_halves>();

        cb_pop_front(cb_input_tiles, 1);

        // Sub-phase B: sum the four Lanczos terms + 3*log(pi) -> cb_output_tiles.
        sum_and_add_const();
    }
}
