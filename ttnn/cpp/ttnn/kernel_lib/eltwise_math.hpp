// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_math.hpp
 * @brief Math SFPU op structs for the eltwise chain — Exp, Log, Sqrt, Rsqrt, Power, ...
 *
 * Each op derives from `UnaryOp<Self, Slot>` (CRTP) and supplies static `init()` + `exec_impl()`.
 * Template parameters carry approx / legacy / fast-and-approx mode + DEST slot at compile time.
 * Runtime-param ops (Power, Rpow) override `exec(uint32_t)` directly to capture instance state.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_op_params.hpp"  // Approx, Legacy

namespace compute_kernel_lib {

// ---- Exp ----
template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
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

// ---- Rsqrt ----
template <Approx fast = Approx::Exact, Legacy legacy = Legacy::Off, Dst Slot = Dst::D0>
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

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.inl"
