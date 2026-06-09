// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_predicates.hpp
 * @brief Predicate / comparison-to-zero / classify SFPU op structs.
 *
 * Eqz, Nez, Ltz, Lez, Gtz, Gez, Isinf, Isnan, Isfinite, Isposinf, Isneginf,
 * LogicalNot, plus runtime-param unary comparisons (UnaryEq/Ne/Gt/Ge/Lt/Le).
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// Comparison-to-zero / classify unary predicates.
template <Dst Slot = Dst::D0>
struct Eqz;
template <Dst Slot = Dst::D0>
struct Nez;
template <Dst Slot = Dst::D0>
struct Ltz;
template <Dst Slot = Dst::D0>
struct Lez;
template <Dst Slot = Dst::D0>
struct Gtz;
template <Dst Slot = Dst::D0>
struct Gez;
template <Dst Slot = Dst::D0>
struct Isinf;
template <Dst Slot = Dst::D0>
struct Isnan;
template <Dst Slot = Dst::D0>
struct Isfinite;
template <Dst Slot = Dst::D0>
struct Isposinf;
template <Dst Slot = Dst::D0>
struct Isneginf;
template <DataFormat DF, Dst Slot = Dst::D0>
struct LogicalNot;

// Runtime-param scalar comparisons.
template <Dst Slot = Dst::D0>
struct UnaryEq;
template <Dst Slot = Dst::D0>
struct UnaryNe;
template <Dst Slot = Dst::D0>
struct UnaryGt;
template <Dst Slot = Dst::D0>
struct UnaryGe;
template <Dst Slot = Dst::D0>
struct UnaryLt;
template <Dst Slot = Dst::D0>
struct UnaryLe;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.inl"
