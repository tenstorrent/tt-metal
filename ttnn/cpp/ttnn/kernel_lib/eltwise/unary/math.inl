// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of math.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#ifndef ARCH_QUASAR  // cbrt.h hard-includes ckernel_sfpu_cbrt.h under TRISC_MATH; Quasar has no such SFPU LLK
#include "api/compute/eltwise_unary/cbrt.h"
#endif
#ifndef ARCH_QUASAR  // log1p.h's log1p_tile_init references SfpuType::log1p, absent from Quasar's SfpuType enum
#include "api/compute/eltwise_unary/log1p.h"
#endif
#ifndef ARCH_QUASAR  // rpow.h hard-includes ckernel_sfpu_rpow.h under TRISC_MATH; Quasar has no such SFPU LLK
#include "api/compute/eltwise_unary/rpow.h"
#endif
#include "api/compute/cumsum.h"              // Cumsum
#include "api/compute/compute_kernel_api.h"  // log_tile / log_tile_init / power_tile

namespace compute_kernel_lib {

// ---- Exp ----
template <Approx approx, Dst Slot>
struct Exp : UnaryOp<Exp<approx, Slot>, Slot> {
    static ALWI void init() { exp_tile_init<approx == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { exp_tile<approx == Approx::Fast>(to_u32(Slot) + slot_offset); }
};

// ---- Log ----
// log_tile / log_tile_init are declared for every arch in compute_kernel_api.h; the Quasar SFPU LLK
// (ckernel_sfpu_log.h) landed with #58209.
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

// ---- Rsqrt ----
template <Approx fast, Dst Slot>
struct Rsqrt : UnaryOp<Rsqrt<fast, Slot>, Slot> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        rsqrt_tile<fast == Approx::Fast ? ckernel::RsqrtMode::Fast : ckernel::RsqrtMode::Default>(
            to_u32(Slot) + slot_offset);
    }
};

// ---- Cbrt ----
// cbrt_tile / cbrt_tile_init have no Quasar SFPU LLK (ckernel_sfpu_cbrt.h absent); guard the struct out to
// match the guarded include above.
#ifndef ARCH_QUASAR
template <Dst Slot>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    static ALWI void init() { cbrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { cbrt_tile(to_u32(Slot) + slot_offset); }
};
#endif

// ---- Log1p — fast (approximate) vs exact mode selected by template ----
// SfpuType::log1p is absent from Quasar's SfpuType enum (log1p.h fails to compile there); guard the struct out
// to match the guarded include above.
#ifndef ARCH_QUASAR
template <Approx fast, Dst Slot>
struct Log1p : UnaryOp<Log1p<fast, Slot>, Slot> {
    static ALWI void init() { log1p_tile_init<fast == Approx::Fast>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { log1p_tile<fast == Approx::Fast>(to_u32(Slot) + slot_offset); }
};
#endif

// ---- Power — runtime exponent. ----
// power_tile / power_tile_init are WH/BH-only (not declared in Quasar's compute_kernel_api.h); guard out
// until the LLK is ported (the template body otherwise fails non-dependent name lookup on Quasar).
#ifndef ARCH_QUASAR
template <Dst Slot>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    constexpr explicit Power(uint32_t e) noexcept : exponent(e) {}
    constexpr Power() noexcept : exponent(0) {}
    static ALWI void init() { power_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { power_tile(to_u32(Slot) + slot_offset, exponent); }
};
#endif

// ---- Rpow — base^x, runtime base. ----
// rpow_tile / rpow_tile_init have no Quasar SFPU LLK (ckernel_sfpu_rpow.h absent); guard the struct out to
// match the guarded include above.
#ifndef ARCH_QUASAR
template <Dst Slot>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base;
    constexpr explicit Rpow(uint32_t b) noexcept : base(b) {}
    constexpr Rpow() noexcept : base(0) {}
    static ALWI void init() { rpow_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { rpow_tile(to_u32(Slot) + slot_offset, base); }
};
#endif

// ---- Cumsum — columnwise cumulative sum (in-DEST). ----
// LLK `cumsum_tile(idst, first)` where `first` resets the accumulator for the first row tile.
// Modelled as a unary op with a single bool param `first` — a FIXED instance field applied
// identically on every tile of the walk (exec ignores the tile index). So one Cumsum element is
// correct only when every tile is a fresh row (`first=true`, default); a genuine multi-tile column
// cumsum (first=true on ht=0, false after) is NOT expressible as a single chain element.
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
// power_iterative_tile / power_iterative_tile_init are WH/BH-only (not declared in Quasar's
// compute_kernel_api.h); guard out until the LLK is ported.
#ifndef ARCH_QUASAR
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
#endif

}  // namespace compute_kernel_lib
