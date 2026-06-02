// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_special.hpp
 * @brief Special-function SFPU op structs — Erf / Erfc / Erfinv / I0 / I1 / Lgamma / Digamma /
 *        Polygamma / TanhDerivative.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/erfinv.h"
#include "api/compute/eltwise_unary/i0.h"
#include "api/compute/eltwise_unary/i1.h"
#include "api/compute/eltwise_unary/digamma.h"
#include "api/compute/eltwise_unary/tanh_derivative.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lgamma.h"

namespace compute_kernel_lib {

// Erf / Erfc — fast_and_approx template param.
template <Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<fast, Slot>, Slot> {
    static ALWI void init() { erf_tile_init<fast == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { erf_tile<fast == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

template <Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<Slot>, Slot> {
    static ALWI void init() { erfc_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { erfc_tile(to_u32(Slot) + slot_offset); }
};

#define ELTWISE_DECLARE_UNARY(Name, fn)                                                             \
    template <Dst Slot = Dst::D0>                                                                   \
    struct Name : UnaryOp<Name<Slot>, Slot> {                                                       \
        static ALWI void init() { fn##_tile_init(); }                                               \
        static ALWI void exec_impl(uint32_t slot_offset) { fn##_tile(to_u32(Slot) + slot_offset); } \
    };

ELTWISE_DECLARE_UNARY(Erfinv, erfinv)
ELTWISE_DECLARE_UNARY(I0, i0)
ELTWISE_DECLARE_UNARY(I1, i1)
ELTWISE_DECLARE_UNARY(Digamma, digamma)
// TanhDerivative lives in eltwise_activations.hpp now (it's an activation
// derivative). Keep tanh_derivative.h include here for backward compat with
// any old call sites that haven't moved over yet.

#undef ELTWISE_DECLARE_UNARY

// Where — ternary y = where(cond, a, b). DEST-only chain element with compile-time
// slot binding. Skips TernaryOp CRTP's distinctness assert (out may equal a/b).
template <DataFormat DF, Dst Cond = Dst::D0, Dst A = Dst::D1, Dst B = Dst::D2, Dst Out = Dst::D0>
struct Where : DestOnlyTag {
    static constexpr uint32_t lane_width = []() {
        uint32_t m = to_u32(Cond);
        if (to_u32(A) > m) {
            m = to_u32(A);
        }
        if (to_u32(B) > m) {
            m = to_u32(B);
        }
        if (to_u32(Out) > m) {
            m = to_u32(Out);
        }
        return m + 1;
    }();
    static ALWI void init() { where_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        where_tile<DF>(
            to_u32(Cond) + slot_offset, to_u32(A) + slot_offset, to_u32(B) + slot_offset, to_u32(Out) + slot_offset);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { exec_impl(slot_offset); }
};

// ---------------------------------------------------------------------------
// lgamma family (DEST-internal SFPU). Three forms used by the lgamma kernels:
//   - LgammaStirling      unary   : lgamma_stirling_tile(idst)
//   - LgammaStirlingFloat binary  : lgamma_stirling_float_tile(x, log_z, out)
//   - LgammaAdjusted      4-slot  : lgamma_adjusted_tile(stirling, logsin, x, out)
// ---------------------------------------------------------------------------

template <Dst Slot = Dst::D0>
struct LgammaStirling : UnaryOp<LgammaStirling<Slot>, Slot> {
    static ALWI void init() { lgamma_stirling_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { lgamma_stirling_tile(to_u32(Slot) + slot_offset); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LgammaStirlingFloat : BinaryOp<LgammaStirlingFloat<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { lgamma_stirling_float_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        lgamma_stirling_float_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// lgamma_adjusted_tile takes 4 explicit DEST args (stirling, logsin, x, out); Out
// may alias an input (the kernel uses (D0,D1,D2,D0)). Hand-rolled DestOnlyTag like
// Where so the 4th arg is the output slot directly (no CRTP single-Out shape).
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct LgammaAdjusted : DestOnlyTag {
    static constexpr uint32_t lane_width = []() {
        uint32_t m = to_u32(In0);
        if (to_u32(In1) > m) {
            m = to_u32(In1);
        }
        if (to_u32(In2) > m) {
            m = to_u32(In2);
        }
        if (to_u32(Out) > m) {
            m = to_u32(Out);
        }
        return m + 1;
    }();
    static ALWI void init() { lgamma_adjusted_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        lgamma_adjusted_tile(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(In2) + slot_offset, to_u32(Out) + slot_offset);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { exec_impl(slot_offset); }
};

}  // namespace compute_kernel_lib
