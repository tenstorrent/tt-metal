// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_math.hpp
 * @brief Tier 1 math ops: Exp, Log, Sqrt, Rsqrt, Recip, Abs, Neg, Square.
 *
 * Each op struct is a 4-line CRTP derivative: declare templates, define
 * `init()` and `call(uint32_t)`. Adding a Tier 2 op is mechanical — see
 * the pattern below.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Approx A = Approx::Exact, Approx Fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp : UnaryOp<Exp<A, Fast, Slot>, Slot> {
    ALWI void init() const { exp_tile_init<static_cast<bool>(A), static_cast<bool>(Fast)>(); }
    ALWI void call(uint32_t d) const { exp_tile<static_cast<bool>(A), static_cast<bool>(Fast)>(d); }
};

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Log : UnaryOp<Log<A, Slot>, Slot> {
    ALWI void init() const { log_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { log_tile<static_cast<bool>(A)>(d); }
};

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : UnaryOp<Sqrt<A, Slot>, Slot> {
    ALWI void init() const { sqrt_tile_init(); }
    ALWI void call(uint32_t d) const { sqrt_tile<static_cast<bool>(A)>(d); }
};

template <Legacy L = Legacy::Off, Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt : UnaryOp<Rsqrt<L, A, Slot>, Slot> {
    ALWI void init() const { rsqrt_tile_init<static_cast<bool>(L)>(); }
    ALWI void call(uint32_t d) const { rsqrt_tile<static_cast<bool>(L), static_cast<bool>(A)>(d); }
};

template <Legacy L = Legacy::On, Dst Slot = Dst::D0>
struct Recip : UnaryOp<Recip<L, Slot>, Slot> {
    ALWI void init() const { recip_tile_init<static_cast<bool>(L)>(); }
    ALWI void call(uint32_t d) const { recip_tile<static_cast<bool>(L)>(d); }
};

template <Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    ALWI void init() const { abs_tile_init(); }
    ALWI void call(uint32_t d) const { abs_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Neg : UnaryOp<Neg<Slot>, Slot> {
    ALWI void init() const { negative_tile_init(); }
    ALWI void call(uint32_t d) const { negative_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Square : UnaryOp<Square<Slot>, Slot> {
    ALWI void init() const { square_tile_init(); }
    ALWI void call(uint32_t d) const { square_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    ALWI void init() const { cbrt_tile_init(); }
    ALWI void call(uint32_t d) const { cbrt_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    ALWI void init() const { sign_tile_init(); }
    ALWI void call(uint32_t d) const { sign_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Signbit : UnaryOp<Signbit<Slot>, Slot> {
    ALWI void init() const { signbit_tile_init(); }
    ALWI void call(uint32_t d) const { signbit_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Exp2 : UnaryOp<Exp2<Slot>, Slot> {
    ALWI void init() const { exp2_tile_init(); }
    ALWI void call(uint32_t d) const { exp2_tile(d); }
};

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Expm1 : UnaryOp<Expm1<A, Slot>, Slot> {
    ALWI void init() const { expm1_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { expm1_tile<static_cast<bool>(A)>(d); }
};

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Log1p : UnaryOp<Log1p<A, Slot>, Slot> {
    ALWI void init() const { log1p_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { log1p_tile<static_cast<bool>(A)>(d); }
};

/// Power(x, e) where `exponent` is a runtime int (uint32_t representation).
template <Dst Slot = Dst::D0>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    ALWI void init() const { power_tile_init(); }
    ALWI void call(uint32_t d) const { power_tile(d, exponent); }
};

/// rpow(base, x) — runtime base, compile-time slot.
template <Dst Slot = Dst::D0>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base_val;
    ALWI void init() const { rpow_tile_init(); }
    ALWI void call(uint32_t d) const { rpow_tile(d, base_val); }
};

template <Dst Slot = Dst::D0>
struct LogWithBase : UnaryOp<LogWithBase<Slot>, Slot> {
    uint32_t base_scale;
    ALWI void init() const { log_with_base_tile_init(); }
    ALWI void call(uint32_t d) const { log_with_base_tile(d, base_scale); }
};

}  // namespace compute_kernel_lib::eltwise
