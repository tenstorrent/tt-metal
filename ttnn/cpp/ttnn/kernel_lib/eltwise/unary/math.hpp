// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file math.hpp
 * @brief Math SFPU op structs for the eltwise chain — Exp, Log, Sqrt, Rsqrt, Power, ...
 *
 * Each op derives from `UnaryOp<Self, Slot>` (CRTP) and supplies static `init()` + `exec_impl()`.
 * Template parameters carry approximation mode + DEST slot at compile time.
 * Runtime-param ops (Power, Rpow) override `exec(uint32_t)` directly to capture instance state.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/op_params.hpp"  // Approx
#ifndef ARCH_QUASAR
#include "sfpu_mode.h"  // ckernel::SfpuMode
#endif

namespace compute_kernel_lib {

// ---- Exp ----
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Exp;

// ---- Log ----
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct Log;

// ---- Sqrt ----
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt;

// ---- Recip (1/x) ----
template <Dst Slot = Dst::D0>
struct Recip;
#ifndef ARCH_QUASAR
// Recip with an explicit SFPU instruction sequence; see recip_tile<SfpuMode> for the per-mode contract.
template <ckernel::SfpuMode MODE, Dst Slot = Dst::D0>
struct RecipMode;
#endif

// ---- Rsqrt ----
template <Approx fast = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt;

// ---- Cbrt ----
template <Dst Slot = Dst::D0>
struct Cbrt;

// ---- Log1p ----
template <Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Log1p;

// ---- Power ----
template <Dst Slot = Dst::D0>
struct Power;

// ---- Rpow ----
template <Dst Slot = Dst::D0>
struct Rpow;

// ---- Cumsum ----
template <Dst Slot = Dst::D0>
struct Cumsum;

// ---- PowerIterative ----
template <Dst Slot = Dst::D0>
struct PowerIterative;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.inl"
