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
// Note: lgamma uses `lgamma_stirling_tile` (multi-tile) — wrap separately when needed.

namespace compute_kernel_lib {

// Erf / Erfc — fast_and_approx template param.
template <Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<fast, Slot>, Slot> {
    static ALWI void init() { erf_tile_init<fast == Approx::Fast>(); }
    static ALWI void exec_impl() { erf_tile<fast == Approx::Fast>(to_u32(Slot)); }
};

template <Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<Slot>, Slot> {
    static ALWI void init() { erfc_tile_init(); }
    static ALWI void exec_impl() { erfc_tile(to_u32(Slot)); }
};

#define ELTWISE_DECLARE_UNARY(Name, fn)                           \
    template <Dst Slot = Dst::D0>                                 \
    struct Name : UnaryOp<Name<Slot>, Slot> {                     \
        static ALWI void init() { fn##_tile_init(); }             \
        static ALWI void exec_impl() { fn##_tile(to_u32(Slot)); } \
    };

ELTWISE_DECLARE_UNARY(Erfinv, erfinv)
ELTWISE_DECLARE_UNARY(I0, i0)
ELTWISE_DECLARE_UNARY(I1, i1)
ELTWISE_DECLARE_UNARY(Digamma, digamma)
ELTWISE_DECLARE_UNARY(TanhDerivative, tanh_derivative)

#undef ELTWISE_DECLARE_UNARY

// Where — ternary y = where(cond, a, b). DEST-only chain element with compile-time
// slot binding. Skips TernaryOp CRTP's distinctness assert (out may equal a/b).
template <DataFormat DF, Dst Cond = Dst::D0, Dst A = Dst::D1, Dst B = Dst::D2, Dst Out = Dst::D0>
struct Where : DestOnlyTag {
    static ALWI void init() { where_tile_init(); }
    static ALWI void exec_impl() { where_tile<DF>(to_u32(Cond), to_u32(A), to_u32(B), to_u32(Out)); }
    ALWI void exec(uint32_t /*i*/) const { exec_impl(); }
    static ALWI void apply() {
        init();
        exec_impl();
    }
};

}  // namespace compute_kernel_lib
