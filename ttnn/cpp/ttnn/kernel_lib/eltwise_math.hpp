// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/compute_kernel_api.h"  // abs_tile{,_init}
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_math.hpp
 * @brief Math op structs for the V2 eltwise helper family.
 *
 * Each op struct derives from `UnaryOp` (CRTP) and exposes `init()` + `call()`.
 * The compute_kernel_api/eltwise_unary headers (exp.h, sqrt.h, recip.h,
 * rsqrt.h) provide the canonical LLK wrappers — `MATH((llk_math_eltwise_unary_sfpu_*))`
 * with `APPROX` and `DST_ACCUM_MODE` injected from the JIT defines. Globally,
 * the user-visible knobs that callers actually flip on a per-op basis are
 * captured as template params here.
 *
 * NOTE: this file does NOT include `sfpu_helpers.hpp`. Calls go directly into
 * `compute_kernel_api/eltwise_unary/...h` and `compute_kernel_api.h`.
 */

// Additional headers for ops below
#include "api/compute/eltwise_unary/negative.h"

namespace compute_kernel_lib {

// =============================================================================
// Negative — flips sign. Trivial init (SFPU type register only).
// =============================================================================

template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Negative : UnaryOp<Negative<ApproxMode, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::negative_tile_init(); }
    ALWI static void call(uint32_t dst) { ckernel::negative_tile(dst); }
};

}  // namespace compute_kernel_lib

namespace compute_kernel_lib {

// =============================================================================
// Abs — S1 (single-param surface: Slot only). No approx variant on the LLK side.
// =============================================================================

/**
 * @brief Element-wise absolute value.
 *
 * Template params:
 *   - `Slot` — DEST slot the op reads/writes (default D0).
 *
 * Underlying LLK: `abs_tile_init` / `abs_tile` from compute_kernel_api.h
 * (which dispatches to `llk_math_eltwise_unary_sfpu_abs_init` /
 * `llk_math_eltwise_unary_sfpu_abs<APPROX>`).
 */
template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<ApproxMode, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::abs_tile_init(); }
    ALWI static void call(uint32_t dst) { ckernel::abs_tile(dst); }
};

// =============================================================================
// Sqrt — S1.
// =============================================================================

template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : UnaryOp<Sqrt<ApproxMode, Slot>, Slot> {
    // sqrt LUT is idempotent — can coexist with another LUT op.
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::sqrt_tile_init(); }
    ALWI static void call(uint32_t dst) {
        // FAST_APPROX template flag on sqrt_tile() corresponds to user-facing
        // Fast/Exact mode.
        if constexpr (ApproxMode == Approx::Fast) {
            ckernel::sqrt_tile<true>(dst);
        } else {
            ckernel::sqrt_tile<false>(dst);
        }
    }
};

// =============================================================================
// Exp — S2 (Approx + FastApprox + Fp32 + Slot). LUT-mutating.
// =============================================================================

/**
 * @brief Element-wise e^x.
 *
 * Template params:
 *   - `ApproxMode` — high-level user-facing Exact / Fast (alias for the
 *     compute_kernel_api `approx` flag on exp_tile_init / exp_tile).
 *   - `FastApprox` — kept for API parity with proposal §3.2 / lessons §1.3.
 *     On the LLK side exp does not split init from exec for this knob; the
 *     `approx` flag on exp_tile_init controls both. This template kept here
 *     so future overloads (e.g. `exp_tile_fast_init` if it lands) can route
 *     orthogonally — for now FastApprox is folded into the same flag.
 *   - `Fp32` — kept for API parity. The LLK reads `DST_ACCUM_MODE` from JIT
 *     defines automatically.
 *   - `Slot` — DEST slot.
 *
 * `clobbers_sfpu_lut = true` — exp programs the SFPU LUT; chaining with
 * another LUT-clobbering op (log, tanh, sigmoid, etc.) disables hoisting.
 */
template <
    Approx ApproxMode = Approx::Exact,
    Approx FastApprox = Approx::Fast,
    FP32DestAcc Fp32 = FP32DestAcc::Off,
    Dst Slot = Dst::D0>
struct Exp : UnaryOp<Exp<ApproxMode, FastApprox, Fp32, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() {
        // exp_tile_init template is <bool approx, uint32_t scale, InputClamping>.
        // User-facing ApproxMode == Fast routes to approx=true.
        constexpr bool approx_v = (ApproxMode == Approx::Fast) || (FastApprox == Approx::Fast);
        ckernel::exp_tile_init<approx_v, 0x3F800000, ckernel::InputClamping::ClampToNegative>();
    }
    ALWI static void call(uint32_t dst) {
        constexpr bool approx_v = (ApproxMode == Approx::Fast) || (FastApprox == Approx::Fast);
        // Default: scale_en=false, ClampToNegative, iterations=8.
        ckernel::exp_tile<approx_v, false, ckernel::InputClamping::ClampToNegative, 8>(dst);
    }
};

// =============================================================================
// Log — S2. LUT-mutating.
// =============================================================================

template <
    Approx ApproxMode = Approx::Exact,
    Approx FastApprox = Approx::Fast,
    FP32DestAcc Fp32 = FP32DestAcc::Off,
    Dst Slot = Dst::D0>
struct Log : UnaryOp<Log<ApproxMode, FastApprox, Fp32, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() {
        constexpr bool fast_v = (ApproxMode == Approx::Fast) || (FastApprox == Approx::Fast);
        ckernel::log_tile_init<fast_v>();
    }
    ALWI static void call(uint32_t dst) {
        constexpr bool fast_v = (ApproxMode == Approx::Fast) || (FastApprox == Approx::Fast);
        ckernel::log_tile<fast_v>(dst);
    }
};

// =============================================================================
// Recip — S2.
// =============================================================================

template <Approx ApproxMode = Approx::Exact, FP32DestAcc Fp32 = FP32DestAcc::Off, Dst Slot = Dst::D0>
struct Recip : UnaryOp<Recip<ApproxMode, Fp32, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() {
        // recip's user knob is `legacy_compat` (default true). Approx::Exact
        // (default) maps to legacy_compat=true (matches prior behavior).
        ckernel::recip_tile_init<true>();
    }
    ALWI static void call(uint32_t dst) { ckernel::recip_tile<true>(dst); }
};

// =============================================================================
// Rsqrt — S8. Full 4-param surface (proposal §3.3).
// =============================================================================

/**
 * @brief Element-wise 1/sqrt(x).
 *
 * Template params (in proposal-spec order):
 *   - `ApproxMode` — routed to init AND exec.
 *   - `Fp32` — exec only on the LLK side (init does not consume it). Kept
 *     in the surface for API parity / future expansion.
 *   - `FastApprox` — exec only on the LLK side. Kept in the surface.
 *   - `LegacyMode` — routed to init AND exec.
 *   - `Slot` — DEST slot.
 *
 * `clobbers_sfpu_lut = false` — rsqrt's LUT is idempotent (verified — see
 * proposal §2.8a / open question 3 resolution). Hoist-safe.
 */
template <
    Approx ApproxMode = Approx::Exact,
    FP32DestAcc Fp32 = FP32DestAcc::Off,
    Approx FastApprox = Approx::Fast,
    Legacy LegacyMode = Legacy::Off,
    Dst Slot = Dst::D0>
struct Rsqrt : UnaryOp<Rsqrt<ApproxMode, Fp32, FastApprox, LegacyMode, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() {
        // rsqrt_tile_init<legacy_compat>(): single param. APPROX is a global
        // define routed inside the LLK wrapper.
        constexpr bool legacy_v = (LegacyMode == Legacy::On);
        ckernel::rsqrt_tile_init<legacy_v>();
    }
    ALWI static void call(uint32_t dst) {
        constexpr bool legacy_v = (LegacyMode == Legacy::On);
        constexpr bool fast_v = (FastApprox == Approx::Fast);
        ckernel::rsqrt_tile<legacy_v, fast_v>(dst);
    }
};

}  // namespace compute_kernel_lib
