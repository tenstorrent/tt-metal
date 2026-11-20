// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file activations.hpp
 * @brief Activation SFPU op structs for the eltwise chain.
 *
 * Wraps `relu_tile`, `sigmoid_tile`, `tanh_tile`, `gelu_tile`, `elu_tile`, `selu_tile`,
 * `hardsigmoid_tile`, `hardtanh_tile`, `hardmish_tile`, `softsign_tile`, `softplus_tile`,
 * `prelu_tile`, `mish_tile`, `silu_tile`, etc.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/op_params.hpp"  // Approx

namespace compute_kernel_lib {

// Simple unary activations.
template <Dst Slot = Dst::D0>
struct Relu;
/// Sigmoid. `fast` selects the LLK's approximate mode. `SigmoidPrecision::Bf16` takes the BF16-accurate path
/// (BF16-accurate exp and one reciprocal step) even with an FP32 DEST, for outputs that keep only BF16 precision; the
/// default, `SigmoidPrecision::Dest`, follows the DEST format as `sigmoid_tile` does.
enum class SigmoidPrecision : uint8_t { Dest, Bf16 };
template <Dst Slot = Dst::D0, Approx fast = Approx::Exact, SigmoidPrecision Precision = SigmoidPrecision::Dest>
struct Sigmoid;
template <Dst Slot = Dst::D0>
struct Tanh;
template <Dst Slot = Dst::D0>
struct Gelu;

// Derivatives — Approx::Exact (default) / Approx::Fast.
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct TanhDerivative;
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct GeluDerivative;

// More unary activations.
template <Dst Slot = Dst::D0>
struct Logsigmoid;
template <Dst Slot = Dst::D0>
struct Hardsigmoid;
template <Dst Slot = Dst::D0>
struct Softsign;
template <Dst Slot = Dst::D0>
struct Hardmish;

// Activations with runtime parameters.
template <Dst Slot = Dst::D0>
struct Hardtanh;
template <Dst Slot = Dst::D0>
struct Elu;
template <Dst Slot = Dst::D0>
struct Selu;
template <Dst Slot = Dst::D0>
struct Softplus;
template <Dst Slot = Dst::D0>
struct Prelu;
template <Dst Slot = Dst::D0>
struct LeakyRelu;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.inl"
