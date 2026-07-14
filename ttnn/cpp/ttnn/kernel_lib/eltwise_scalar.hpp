// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_scalar.hpp
 * @brief Scalar-parameter SFPU op structs — Threshold, Clamp, AddUnary, SubUnary, MulUnary,
 *        DivUnary, RsubUnary, RdivUnary, Dropout.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// Threshold — runtime threshold + value.
template <Dst Slot = Dst::D0>
struct Threshold;

// Clamp — runtime min/max.
template <Dst Slot = Dst::D0>
struct Clamp;

// X-macro-generated binop-with-scalar wrappers.
template <Dst Slot = Dst::D0>
struct AddUnary;

template <Dst Slot = Dst::D0>
struct SubUnary;

template <Dst Slot = Dst::D0>
struct MulUnary;

template <Dst Slot = Dst::D0>
struct DivUnary;

template <Dst Slot = Dst::D0>
struct RsubUnary;

// Dropout — runtime probability + scale_factor.
template <Dst Slot = Dst::D0>
struct Dropout;

// RdivUnary — own header with its own init.
template <Dst Slot = Dst::D0>
struct RdivUnary;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.inl"
