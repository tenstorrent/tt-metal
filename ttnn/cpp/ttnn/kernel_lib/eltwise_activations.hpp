// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_activations.hpp
 * @brief Activation SFPU op structs for the eltwise chain.
 *
 * Wraps `relu_tile`, `sigmoid_tile`, `tanh_tile`, `gelu_tile`, `elu_tile`, `selu_tile`,
 * `hardsigmoid_tile`, `hardtanh_tile`, `hardmish_tile`, `softsign_tile`, `softplus_tile`,
 * `prelu_tile`, `mish_tile`, `silu_tile`, etc.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#include "api/compute/compute_kernel_api.h"         // sigmoid_tile, tanh_tile (also gelu fallback)
#include "api/compute/eltwise_unary/activations.h"  // hardsigmoid, softsign, softshrink, hardshrink, celu
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/prelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace compute_kernel_lib {

template <Dst Slot = Dst::D0>
struct Relu : UnaryOp<Relu<Slot>, Slot> {
    static ALWI void init() { relu_tile_init(); }
    static ALWI void call(uint32_t idst) { relu_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Sigmoid : UnaryOp<Sigmoid<Slot>, Slot> {
    static ALWI void init() { sigmoid_tile_init(); }
    static ALWI void call(uint32_t idst) { sigmoid_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Tanh : UnaryOp<Tanh<Slot>, Slot> {
    static ALWI void init() { tanh_tile_init(); }
    static ALWI void call(uint32_t idst) { tanh_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Gelu : UnaryOp<Gelu<Slot>, Slot> {
    static ALWI void init() { gelu_tile_init(); }
    static ALWI void call(uint32_t idst) { gelu_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Hardsigmoid : UnaryOp<Hardsigmoid<Slot>, Slot> {
    static ALWI void init() { hardsigmoid_tile_init(); }
    static ALWI void call(uint32_t idst) { hardsigmoid_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Softsign : UnaryOp<Softsign<Slot>, Slot> {
    static ALWI void init() { softsign_tile_init(); }
    static ALWI void call(uint32_t idst) { softsign_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Hardmish : UnaryOp<Hardmish<Slot>, Slot> {
    static ALWI void init() { hardmish_tile_init(); }
    static ALWI void call(uint32_t idst) { hardmish_tile(idst); }
};

// Hardtanh — runtime min/max via ctor.
template <Dst Slot = Dst::D0>
struct Hardtanh : UnaryOp<Hardtanh<Slot>, Slot> {
    uint32_t min_param;
    uint32_t max_param;
    constexpr Hardtanh(uint32_t lo, uint32_t hi) noexcept : min_param(lo), max_param(hi) {}
    constexpr Hardtanh() noexcept : min_param(0), max_param(0) {}

    static ALWI void init() { hardtanh_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) { /* runtime params — exec uses members */ }

    ALWI void exec(uint32_t /*i*/) const { hardtanh_tile(to_u32(Slot), min_param, max_param); }
};

// Elu — runtime alpha.
template <Dst Slot = Dst::D0>
struct Elu : UnaryOp<Elu<Slot>, Slot> {
    uint32_t alpha;
    constexpr explicit Elu(uint32_t a) noexcept : alpha(a) {}
    constexpr Elu() noexcept : alpha(0) {}

    static ALWI void init() { elu_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { elu_tile(to_u32(Slot), alpha); }
};

template <Dst Slot = Dst::D0>
struct Selu : UnaryOp<Selu<Slot>, Slot> {
    uint32_t scale;
    uint32_t alpha;
    constexpr Selu(uint32_t s, uint32_t a) noexcept : scale(s), alpha(a) {}
    constexpr Selu() noexcept : scale(0), alpha(0) {}
    static ALWI void init() { selu_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { selu_tile(to_u32(Slot), scale, alpha); }
};

template <Dst Slot = Dst::D0>
struct Softplus : UnaryOp<Softplus<Slot>, Slot> {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    constexpr Softplus(uint32_t b, uint32_t br, uint32_t t) noexcept : beta(b), beta_recip(br), threshold(t) {}
    constexpr Softplus() noexcept : beta(0), beta_recip(0), threshold(0) {}
    static ALWI void init() { softplus_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { softplus_tile(to_u32(Slot), beta, beta_recip, threshold); }
};

template <Dst Slot = Dst::D0>
struct Prelu : UnaryOp<Prelu<Slot>, Slot> {
    uint32_t param0;
    constexpr explicit Prelu(uint32_t p) noexcept : param0(p) {}
    constexpr Prelu() noexcept : param0(0) {}
    static ALWI void init() { prelu_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { prelu_tile(to_u32(Slot), param0); }
};

// LeakyRelu — runtime slope.
template <Dst Slot = Dst::D0>
struct LeakyRelu : UnaryOp<LeakyRelu<Slot>, Slot> {
    uint32_t slope;
    constexpr explicit LeakyRelu(uint32_t s) noexcept : slope(s) {}
    constexpr LeakyRelu() noexcept : slope(0) {}
    static ALWI void init() { leaky_relu_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { leaky_relu_tile(to_u32(Slot), slope); }
};

}  // namespace compute_kernel_lib
