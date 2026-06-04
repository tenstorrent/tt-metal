// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_ternary.hpp
 * @brief Chain-family ternary SFPU op structs — Lerp, SnakeBeta (static) and
 *        Addcmul, Addcdiv (runtime scalar `value`).
 *
 * These are the `eltwise_chain` family (DestOnlyTag + `exec(i, slot_offset)`),
 * NOT the `SfpuChain` combinator family of the same names in `sfpu_helpers.hpp`
 * (which expose `call(a,b,c,d)` and cannot be passed to `eltwise_chain`). All
 * four wrap a 4-DEST-arg LLK `op_tile<DF>(in0, in1, in2, out)`; addcmul/addcdiv
 * additionally take a runtime `value` (uint32 bits).
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// Chain-family ternary SFPU op structs (definitions in eltwise_ternary.inl).
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct Lerp;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct SnakeBeta;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct Addcmul;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct Addcdiv;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_ternary.inl"
