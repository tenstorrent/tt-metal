// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/xielu.h"
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/threshold.h"
#include "api/compute/eltwise_unary/prelu.h"

namespace compute_kernel_lib {

// --- Activations ---

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sigmoid : UnaryOp<Sigmoid<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Tanh : UnaryOp<Tanh<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Gelu : UnaryOp<Gelu<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/**
 * @brief GELU derivative for backward pass: d/dx gelu(x).
 *
 * Wraps gelu_derivative_tile_init<fast>() and gelu_derivative_tile<fast>(d0).
 * Default Approx::Exact uses the piecewise polynomial (Max ULP = 1 on BF16);
 * Approx::Fast uses the faster formula-based kernel.
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct GeluDerivative : UnaryOp<GeluDerivative<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Silu : UnaryOp<Silu<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Relu : UnaryOp<Relu<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardmish : UnaryOp<Hardmish<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardsigmoid : UnaryOp<Hardsigmoid<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardtanh : UnaryOp<Hardtanh<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softsign : UnaryOp<Softsign<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softplus : UnaryOp<Softplus<Slot>, Slot> {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Xielu : UnaryOp<Xielu<Slot>, Slot> {
    uint32_t alpha_p;
    uint32_t alpha_n;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Additional Activations ---

template <Dst Slot = Dst::D0>
struct Elu : UnaryOp<Elu<Slot>, Slot> {
    uint32_t alpha;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Selu : UnaryOp<Selu<Slot>, Slot> {
    uint32_t scale;
    uint32_t alpha;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Celu : UnaryOp<Celu<Slot>, Slot> {
    uint32_t alpha;
    uint32_t alpha_recip;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softshrink : UnaryOp<Softshrink<Slot>, Slot> {
    uint32_t lambda;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Clamp : UnaryOp<Clamp<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Threshold : UnaryOp<Threshold<Slot>, Slot> {
    uint32_t threshold;
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Prelu : UnaryOp<Prelu<Slot>, Slot> {
    uint32_t weight;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Activation aliases ---
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_activations.inl"
