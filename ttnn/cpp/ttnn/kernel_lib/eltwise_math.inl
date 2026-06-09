// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_math.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/cumsum.h"              // Cumsum
#include "api/compute/compute_kernel_api.h"  // log_tile / log_tile_init / power_tile

namespace compute_kernel_lib {

// ---- Exp ----
// The second `fast` template parameter is part of the struct signature but is NOT
// routed into the LLK template args. The LLK `exp_tile_init` second arg is
// `uint32_t scale` (default 0x3F800000 = 1.0f) and `exp_tile` second arg is
// `bool scale_en` (default false). Only `approx` is forwarded to the LLK; the
// scale and scale_en arguments stay at their LLK defaults.
template <Approx approx, Approx fast, Dst Slot>
struct Exp : UnaryOp<Exp<approx, fast, Slot>, Slot> {
    static ALWI void init() { exp_tile_init<approx == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { exp_tile<approx == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

// ---- Log ----
template <Approx fast, Dst Slot>
struct Log : UnaryOp<Log<fast, Slot>, Slot> {
    static ALWI void init() { log_tile_init<fast == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { log_tile<fast == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

// ---- Sqrt ----
template <Approx fast, Dst Slot>
struct Sqrt : UnaryOp<Sqrt<fast, Slot>, Slot> {
    static ALWI void init() { sqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { sqrt_tile<fast == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

// ---- Recip (1/x) ----
template <Dst Slot>
struct Recip : UnaryOp<Recip<Slot>, Slot> {
    static ALWI void init() { recip_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { recip_tile(to_u32(Slot) + slot_offset); }
};

// ---- Rsqrt — Approx (Fast/Exact) and Legacy (On/Off). Templated on both.
template <Approx fast, Legacy legacy, Dst Slot>
struct Rsqrt : UnaryOp<Rsqrt<fast, legacy, Slot>, Slot> {
    static ALWI void init() { rsqrt_tile_init<legacy == Legacy::On>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        rsqrt_tile<legacy == Legacy::On, fast == Approx::Fast>(to_u32(Slot) + slot_offset);
    }
};

// ---- Cbrt ----
template <Dst Slot>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    static ALWI void init() { cbrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { cbrt_tile(to_u32(Slot) + slot_offset); }
};

// ---- Log1p — fast (approximate) vs exact mode selected by template ----
template <Approx fast, Dst Slot>
struct Log1p : UnaryOp<Log1p<fast, Slot>, Slot> {
    static ALWI void init() { log1p_tile_init<fast == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { log1p_tile<fast == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

// ---- Power — runtime exponent. ----
template <Dst Slot>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    constexpr explicit Power(uint32_t e) noexcept : exponent(e) {}
    constexpr Power() noexcept : exponent(0) {}
    static ALWI void init() { power_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { power_tile(to_u32(Slot) + slot_offset, exponent); }
};

// ---- Rpow — base^x, runtime base. ----
template <Dst Slot>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base;
    constexpr explicit Rpow(uint32_t b) noexcept : base(b) {}
    constexpr Rpow() noexcept : base(0) {}
    static ALWI void init() { rpow_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { rpow_tile(to_u32(Slot) + slot_offset, base); }
};

// ---- Cumsum — columnwise cumulative sum (in-DEST). ----
// LLK `cumsum_tile(idst, first)` where `first` resets the accumulator for the
// first row tile. Modelled as a unary op with a single bool param `first` so
// the chain can iterate over consecutive tiles with `first=true` only on the
// initial call. For chain-driven multi-tile loops where each iteration is a
// fresh row, leave `first = true` (default).
template <Dst Slot>
struct Cumsum : UnaryOp<Cumsum<Slot>, Slot> {
    bool first;
    constexpr explicit Cumsum(bool f = true) noexcept : first(f) {}
    static ALWI void init() { cumsum_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { cumsum_tile(to_u32(Slot) + slot_offset, first); }
};

// ---- PowerIterative — positive-integer exponent via iterative multiply. ----
// Distinct LLK from Power: power_iterative_tile uses an iterative loop; faster for
// small integer exponents. Only supports positive integer scalars.
template <Dst Slot>
struct PowerIterative : UnaryOp<PowerIterative<Slot>, Slot> {
    uint32_t exponent;
    constexpr explicit PowerIterative(uint32_t e) noexcept : exponent(e) {}
    constexpr PowerIterative() noexcept : exponent(0) {}
    static ALWI void init() { power_iterative_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        power_iterative_tile(to_u32(Slot) + slot_offset, exponent);
    }
};

}  // namespace compute_kernel_lib
