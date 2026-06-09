// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_activations.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/compute_kernel_api.h"         // sigmoid_tile, tanh_tile (also gelu fallback)
#include "api/compute/eltwise_unary/activations.h"  // hardsigmoid, softsign, softshrink, hardshrink, celu
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/tanh_derivative.h"  // TanhDerivative
#include "api/compute/logsigmoid.h"                     // Logsigmoid (binary in-DEST)
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/prelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace compute_kernel_lib {

template <Dst Slot>
struct Relu : UnaryOp<Relu<Slot>, Slot> {
    static ALWI void init() { relu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { relu_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Sigmoid : UnaryOp<Sigmoid<Slot>, Slot> {
    static ALWI void init() { sigmoid_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { sigmoid_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Tanh : UnaryOp<Tanh<Slot>, Slot> {
    static ALWI void init() { tanh_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { tanh_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Gelu : UnaryOp<Gelu<Slot>, Slot> {
    static ALWI void init() { gelu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { gelu_tile(to_u32(Slot) + slot_offset); }
};

/// d/dx(tanh) = sech²(x). LLK uses `fast_and_approx` bool template param;
/// surfaced as `Approx::Exact` (default) / `Approx::Fast` in chain style.
template <Approx fast, Dst Slot>
struct TanhDerivative : UnaryOp<TanhDerivative<fast, Slot>, Slot> {
    static ALWI void init() { tanh_derivative_tile_init<static_cast<bool>(fast)>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        tanh_derivative_tile<static_cast<bool>(fast)>(to_u32(Slot) + slot_offset);
    }
};

/// d/dx(gelu). Same Approx pattern as TanhDerivative.
template <Approx fast, Dst Slot>
struct GeluDerivative : UnaryOp<GeluDerivative<fast, Slot>, Slot> {
    static ALWI void init() { gelu_derivative_tile_init<static_cast<bool>(fast)>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        gelu_derivative_tile<static_cast<bool>(fast)>(to_u32(Slot) + slot_offset);
    }
};

/// Logsigmoid binary-in-DEST: logsigmoid_tile(In0, In1, Out) where caller
/// has loaded x into D[In0] and exp(-x) into D[In1]; result placed in D[Out].
/// Modelled as BinaryOp (3-DEST, no CB sources).
template <Dst In0, Dst In1, Dst Out>
struct Logsigmoid : BinaryOp<Logsigmoid<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { logsigmoid_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        logsigmoid_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst Slot>
struct Hardsigmoid : UnaryOp<Hardsigmoid<Slot>, Slot> {
    static ALWI void init() { hardsigmoid_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { hardsigmoid_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Softsign : UnaryOp<Softsign<Slot>, Slot> {
    static ALWI void init() { softsign_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { softsign_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Hardmish : UnaryOp<Hardmish<Slot>, Slot> {
    static ALWI void init() { hardmish_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { hardmish_tile(to_u32(Slot) + slot_offset); }
};

// Hardtanh — runtime min/max via ctor; overrides exec(uint32_t).
template <Dst Slot>
struct Hardtanh : UnaryOp<Hardtanh<Slot>, Slot> {
    uint32_t min_param;
    uint32_t max_param;
    constexpr Hardtanh(uint32_t lo, uint32_t hi) noexcept : min_param(lo), max_param(hi) {}
    constexpr Hardtanh() noexcept : min_param(0), max_param(0) {}

    static ALWI void init() { hardtanh_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        hardtanh_tile(to_u32(Slot) + slot_offset, min_param, max_param);
    }
};

// Elu — runtime alpha.
template <Dst Slot>
struct Elu : UnaryOp<Elu<Slot>, Slot> {
    uint32_t alpha;
    constexpr explicit Elu(uint32_t a) noexcept : alpha(a) {}
    constexpr Elu() noexcept : alpha(0) {}

    static ALWI void init() { elu_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { elu_tile(to_u32(Slot) + slot_offset, alpha); }
};

template <Dst Slot>
struct Selu : UnaryOp<Selu<Slot>, Slot> {
    uint32_t scale;
    uint32_t alpha;
    constexpr Selu(uint32_t s, uint32_t a) noexcept : scale(s), alpha(a) {}
    constexpr Selu() noexcept : scale(0), alpha(0) {}
    static ALWI void init() { selu_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { selu_tile(to_u32(Slot) + slot_offset, scale, alpha); }
};

template <Dst Slot>
struct Softplus : UnaryOp<Softplus<Slot>, Slot> {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    constexpr Softplus(uint32_t b, uint32_t br, uint32_t t) noexcept : beta(b), beta_recip(br), threshold(t) {}
    constexpr Softplus() noexcept : beta(0), beta_recip(0), threshold(0) {}
    static ALWI void init() { softplus_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        softplus_tile(to_u32(Slot) + slot_offset, beta, beta_recip, threshold);
    }
};

template <Dst Slot>
struct Prelu : UnaryOp<Prelu<Slot>, Slot> {
    uint32_t param0;
    constexpr explicit Prelu(uint32_t p) noexcept : param0(p) {}
    constexpr Prelu() noexcept : param0(0) {}
    static ALWI void init() { prelu_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { prelu_tile(to_u32(Slot) + slot_offset, param0); }
};

// LeakyRelu — runtime slope.
template <Dst Slot>
struct LeakyRelu : UnaryOp<LeakyRelu<Slot>, Slot> {
    uint32_t slope;
    constexpr explicit LeakyRelu(uint32_t s) noexcept : slope(s) {}
    constexpr LeakyRelu() noexcept : slope(0) {}
    static ALWI void init() { leaky_relu_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { leaky_relu_tile(to_u32(Slot) + slot_offset, slope); }
};

}  // namespace compute_kernel_lib
