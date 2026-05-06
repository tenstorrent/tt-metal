// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_math.hpp
 * @brief Math SFPU op structs for the eltwise chain — Exp, Log, Sqrt, Rsqrt, Power, ...
 *
 * Each op derives from `UnaryOp<Self, Slot>` (CRTP) and supplies static `init()` + `call(idst)`.
 * Template parameters carry approx / legacy / fast-and-approx mode + DEST slot at compile time.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/compute_kernel_api.h"  // log_tile / log_tile_init / power_tile

namespace compute_kernel_lib {

// ---- Exp ----
//
// `exp_tile_init<APPROX, FAST_AND_APPROX>()` and `exp_tile<APPROX, FAST_AND_APPROX>(idst, ...)`.
template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp : UnaryOp<Exp<approx, fast, Slot>, Slot> {
    static ALWI void init() { exp_tile_init<approx == Approx::Fast, fast == Approx::Fast>(); }
    static ALWI void call(uint32_t idst) { exp_tile<approx == Approx::Fast, fast == Approx::Fast>(idst); }
};

// ---- Log ----
template <Dst Slot = Dst::D0>
struct Log : UnaryOp<Log<Slot>, Slot> {
    static ALWI void init() { log_tile_init(); }
    static ALWI void call(uint32_t idst) { log_tile(idst); }
};

// ---- Sqrt ----
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : UnaryOp<Sqrt<fast, Slot>, Slot> {
    static ALWI void init() { sqrt_tile_init(); }
    static ALWI void call(uint32_t idst) { sqrt_tile<fast == Approx::Fast>(idst); }
};

// ---- Recip (1/x) ----
template <Dst Slot = Dst::D0>
struct Recip : UnaryOp<Recip<Slot>, Slot> {
    static ALWI void init() { recip_tile_init(); }
    static ALWI void call(uint32_t idst) { recip_tile(idst); }
};

// ---- Rsqrt — Approx (Fast/Exact) and Legacy (On/Off). Templated on both.
template <Approx fast = Approx::Exact, Legacy legacy = Legacy::Off, Dst Slot = Dst::D0>
struct Rsqrt : UnaryOp<Rsqrt<fast, legacy, Slot>, Slot> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void call(uint32_t idst) { rsqrt_tile<fast == Approx::Fast>(idst); }
};

// ---- Cbrt ----
template <Dst Slot = Dst::D0>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    static ALWI void init() { cbrt_tile_init(); }
    static ALWI void call(uint32_t idst) { cbrt_tile(idst); }
};

// ---- Log1p ----
template <Dst Slot = Dst::D0>
struct Log1p : UnaryOp<Log1p<Slot>, Slot> {
    static ALWI void init() { log1p_tile_init(); }
    static ALWI void call(uint32_t idst) { log1p_tile(idst); }
};

// ---- Power — runtime exponent (passes uint32_t param). ----
template <Dst Slot = Dst::D0>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    constexpr explicit Power(uint32_t e) noexcept : exponent(e) {}
    constexpr Power() noexcept : exponent(0) {}
    static ALWI void init() { power_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { power_tile(to_u32(Slot), exponent); }
};

// ---- Rpow — base^x, runtime base. ----
template <Dst Slot = Dst::D0>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base;
    constexpr explicit Rpow(uint32_t b) noexcept : base(b) {}
    constexpr Rpow() noexcept : base(0) {}
    static ALWI void init() { rpow_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { rpow_tile(to_u32(Slot), base); }
};

}  // namespace compute_kernel_lib
