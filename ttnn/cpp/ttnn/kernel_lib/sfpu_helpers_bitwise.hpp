// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Opt-in extension for sfpu_helpers: bitwise and shift operations.
//
// These are split out because the underlying ckernel headers
// (ckernel_sfpu_bitwise_and.h, etc.) contain `using namespace sfpi;`
// at file scope, which introduces operator overloads (e.g. operator&)
// that create ambiguous-overload errors when combined with reduce.h
// or other LLK headers.
//
// Include this header *only* in kernels that need BitwiseAnd / BitwiseOr /
// BitwiseXor / BitwiseNot / LeftShift / RightShift op structs.

#pragma once

#include "sfpu_helpers.hpp"

namespace compute_kernel_lib {

// --- Bitwise Ops ---

template <Dst Slot = Dst::D0>
struct BitwiseNot;
template <Dst Slot = Dst::D0>
struct BitwiseAnd;
template <Dst Slot = Dst::D0>
struct BitwiseOr;
template <Dst Slot = Dst::D0>
struct BitwiseXor;

// --- Shift Ops ---

template <Dst Slot = Dst::D0>
struct LeftShift;
template <Dst Slot = Dst::D0>
struct RightShift;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers_bitwise.inl"
