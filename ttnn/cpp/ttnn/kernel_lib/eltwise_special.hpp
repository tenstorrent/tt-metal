// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_special.hpp
 * @brief Special-function SFPU op structs — Erf / Erfc / Erfinv / I0 / I1 / Lgamma / Digamma /
 *        Polygamma / TanhDerivative.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// Erf / Erfc
template <Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Erf;

template <Dst Slot = Dst::D0>
struct Erfc;

// X-macro-generated unary special functions.
template <Dst Slot = Dst::D0>
struct Erfinv;

template <Dst Slot = Dst::D0>
struct I0;

template <Dst Slot = Dst::D0>
struct I1;

template <Dst Slot = Dst::D0>
struct Digamma;

// Where — ternary y = where(cond, a, b).
template <DataFormat DF, Dst Cond = Dst::D0, Dst A = Dst::D1, Dst B = Dst::D2, Dst Out = Dst::D0>
struct Where;

// lgamma family.
template <Dst Slot = Dst::D0>
struct LgammaStirling;

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LgammaStirlingFloat;

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct LgammaAdjusted;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.inl"
