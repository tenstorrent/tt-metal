// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_scalar.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/threshold.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/dropout.h"  // Dropout

namespace compute_kernel_lib {

// Threshold — runtime threshold + value.
template <Dst Slot>
struct Threshold : UnaryOp<Threshold<Slot>, Slot> {
    uint32_t threshold;
    uint32_t value;
    constexpr Threshold(uint32_t t, uint32_t v) noexcept : threshold(t), value(v) {}
    constexpr Threshold() noexcept : threshold(0), value(0) {}
    static ALWI void init() { threshold_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        threshold_tile(to_u32(Slot) + slot_offset, threshold, value);
    }
};

// Clamp — runtime min/max.
template <Dst Slot>
struct Clamp : UnaryOp<Clamp<Slot>, Slot> {
    uint32_t min_param;
    uint32_t max_param;
    constexpr Clamp(uint32_t lo, uint32_t hi) noexcept : min_param(lo), max_param(hi) {}
    constexpr Clamp() noexcept : min_param(0), max_param(0) {}
    static ALWI void init() { clamp_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        clamp_tile(to_u32(Slot) + slot_offset, min_param, max_param);
    }
};

// Generic binop-with-scalar wrappers — share `binop_with_scalar_tile_init`.
#define ELTWISE_DECLARE_BINOP_SCALAR(Name, fn)                                                                 \
    template <Dst Slot>                                                                                        \
    struct Name : UnaryOp<Name<Slot>, Slot> {                                                                  \
        uint32_t param0;                                                                                       \
        constexpr explicit Name(uint32_t p) noexcept : param0(p) {}                                            \
        constexpr Name() noexcept : param0(0) {}                                                               \
        static ALWI void init() { binop_with_scalar_tile_init(); }                                             \
        ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { fn(to_u32(Slot) + slot_offset, param0); } \
    };

ELTWISE_DECLARE_BINOP_SCALAR(AddUnary, add_unary_tile)
ELTWISE_DECLARE_BINOP_SCALAR(SubUnary, sub_unary_tile)
ELTWISE_DECLARE_BINOP_SCALAR(MulUnary, mul_unary_tile)
ELTWISE_DECLARE_BINOP_SCALAR(DivUnary, div_unary_tile)
ELTWISE_DECLARE_BINOP_SCALAR(RsubUnary, rsub_unary_tile)

// Dropout — runtime probability + scale_factor + seed (all packed u32 bits).
// `dropout_kernel_init(seed)` is dropout's per-op SFPU init (it seeds the PRNG); it's
// the dropout analogue of every other op's `*_tile_init`, except it takes the runtime
// seed. So `init()` here is a normal instance method (non-static, like `exec`) that reads
// the struct's `seed` — the chain dispatches init on the instance, so this just works.
// The chain emits it once (boot-hoist), the once-per-kernel seeding the original kernel
// did out-of-band.
template <Dst Slot>
struct Dropout : UnaryOp<Dropout<Slot>, Slot> {
    uint32_t probability;
    uint32_t scale_factor;
    uint32_t seed;
    constexpr Dropout(uint32_t p, uint32_t s, uint32_t seed_) noexcept :
        probability(p), scale_factor(s), seed(seed_) {}
    constexpr Dropout() noexcept : probability(0), scale_factor(0), seed(0) {}
    ALWI void init() const { dropout_kernel_init(seed); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        dropout_tile(to_u32(Slot) + slot_offset, probability, scale_factor);
    }
};

// rdiv is in its own header with its own init.
template <Dst Slot>
struct RdivUnary : UnaryOp<RdivUnary<Slot>, Slot> {
    uint32_t param0;
    constexpr explicit RdivUnary(uint32_t p) noexcept : param0(p) {}
    constexpr RdivUnary() noexcept : param0(0) {}
    static ALWI void init() { rdiv_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { rdiv_tile(to_u32(Slot) + slot_offset, param0); }
};

#undef ELTWISE_DECLARE_BINOP_SCALAR

}  // namespace compute_kernel_lib
