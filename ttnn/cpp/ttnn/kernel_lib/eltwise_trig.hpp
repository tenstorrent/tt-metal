// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_trig.hpp
 * @brief Trigonometric SFPU op structs.
 *
 * Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// Trigonometric SFPU op structs (definitions in eltwise_trig.inl).
template <Dst Slot = Dst::D0>
struct Sin;
template <Dst Slot = Dst::D0>
struct Cos;
template <Dst Slot = Dst::D0>
struct Tan;
template <Dst Slot = Dst::D0>
struct Asin;
template <Dst Slot = Dst::D0>
struct Acos;
template <Dst Slot = Dst::D0>
struct Atan;
template <Dst Slot = Dst::D0>
struct Sinh;
template <Dst Slot = Dst::D0>
struct Cosh;
template <Dst Slot = Dst::D0>
struct Asinh;
template <Dst Slot = Dst::D0>
struct Acosh;
template <Dst Slot = Dst::D0>
struct Atanh;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.inl"
