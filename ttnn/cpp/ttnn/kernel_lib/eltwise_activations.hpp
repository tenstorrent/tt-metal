// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/threshold.h"
#include "api/compute/eltwise_unary/prelu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/xielu.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_activations.hpp
 * @brief Tier 1 activations: Sigmoid, Tanh, Gelu, Relu.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Sigmoid : UnaryOp<Sigmoid<A, Slot>, Slot> {
    ALWI void init() const { sigmoid_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { sigmoid_tile<(int)VectorMode::RC, static_cast<bool>(A)>(d); }
};

template <Approx A = Approx::Exact, Dst Slot = Dst::D0>
struct Tanh : UnaryOp<Tanh<A, Slot>, Slot> {
    ALWI void init() const { tanh_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { tanh_tile<static_cast<bool>(A)>(d); }
};

template <Approx A = Approx::Fast, Dst Slot = Dst::D0>
struct Gelu : UnaryOp<Gelu<A, Slot>, Slot> {
    ALWI void init() const { gelu_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { gelu_tile<static_cast<bool>(A)>(d); }
};

template <Dst Slot = Dst::D0>
struct Relu : UnaryOp<Relu<Slot>, Slot> {
    ALWI void init() const { relu_tile_init(); }
    ALWI void call(uint32_t d) const { relu_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Silu : UnaryOp<Silu<Slot>, Slot> {
    ALWI void init() const { silu_tile_init(); }
    ALWI void call(uint32_t d) const { silu_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Hardsigmoid : UnaryOp<Hardsigmoid<Slot>, Slot> {
    ALWI void init() const { hardsigmoid_tile_init(); }
    ALWI void call(uint32_t d) const { hardsigmoid_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Hardmish : UnaryOp<Hardmish<Slot>, Slot> {
    ALWI void init() const { hardmish_tile_init(); }
    ALWI void call(uint32_t d) const { hardmish_tile(d); }
};

/// Hardtanh — runtime min / max (uint32_t bit pattern of float).
template <Dst Slot = Dst::D0>
struct Hardtanh : UnaryOp<Hardtanh<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const { hardtanh_tile_init(); }
    ALWI void call(uint32_t d) const { hardtanh_tile(d, param_min, param_max); }
};

template <Dst Slot = Dst::D0>
struct Softsign : UnaryOp<Softsign<Slot>, Slot> {
    ALWI void init() const { softsign_tile_init(); }
    ALWI void call(uint32_t d) const { softsign_tile(d); }
};

/// Softplus — runtime beta / beta_recip / threshold (uint32_t bit patterns).
template <Dst Slot = Dst::D0>
struct Softplus : UnaryOp<Softplus<Slot>, Slot> {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    ALWI void init() const { softplus_tile_init(); }
    ALWI void call(uint32_t d) const { softplus_tile(d, beta, beta_recip, threshold); }
};

/// Xielu — runtime alpha_p / alpha_n.
template <Dst Slot = Dst::D0>
struct Xielu : UnaryOp<Xielu<Slot>, Slot> {
    uint32_t alpha_p;
    uint32_t alpha_n;
    ALWI void init() const { xielu_tile_init(); }
    ALWI void call(uint32_t d) const { xielu_tile(d, alpha_p, alpha_n); }
};

template <Dst Slot = Dst::D0>
struct Elu : UnaryOp<Elu<Slot>, Slot> {
    uint32_t alpha;
    ALWI void init() const { elu_tile_init(); }
    ALWI void call(uint32_t d) const { elu_tile(d, alpha); }
};

template <Dst Slot = Dst::D0>
struct Selu : UnaryOp<Selu<Slot>, Slot> {
    uint32_t scale;
    uint32_t alpha;
    ALWI void init() const { selu_tile_init(); }
    ALWI void call(uint32_t d) const { selu_tile(d, scale, alpha); }
};

template <Dst Slot = Dst::D0>
struct Celu : UnaryOp<Celu<Slot>, Slot> {
    uint32_t alpha;
    uint32_t alpha_recip;
    ALWI void init() const { celu_tile_init(); }
    ALWI void call(uint32_t d) const { celu_tile(d, alpha, alpha_recip); }
};

template <Dst Slot = Dst::D0>
struct Softshrink : UnaryOp<Softshrink<Slot>, Slot> {
    uint32_t lambda;
    ALWI void init() const { softshrink_tile_init(); }
    ALWI void call(uint32_t d) const { softshrink_tile(d, lambda); }
};

template <Dst Slot = Dst::D0>
struct Clamp : UnaryOp<Clamp<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const { clamp_tile_init(); }
    ALWI void call(uint32_t d) const { clamp_tile(d, param_min, param_max); }
};

template <Dst Slot = Dst::D0>
struct Threshold : UnaryOp<Threshold<Slot>, Slot> {
    uint32_t threshold;
    uint32_t value;
    ALWI void init() const { threshold_tile_init(); }
    ALWI void call(uint32_t d) const { threshold_tile(d, threshold, value); }
};

template <Dst Slot = Dst::D0>
struct Prelu : UnaryOp<Prelu<Slot>, Slot> {
    uint32_t weight;
    ALWI void init() const { prelu_tile_init(); }
    ALWI void call(uint32_t d) const { prelu_tile(d, weight); }
};

}  // namespace compute_kernel_lib::eltwise
