// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_convenience.hpp
 * @brief One-liner entry points for the dominant eltwise chain shapes.
 *
 * Each wrapper is a pure inline forwarder to `eltwise_chain` for one common shape, so a
 * simple op needs one call instead of a hand-written chain. The op is baked into the name
 * (`add`/`sub`/`mul`, or the SFPU op as a type parameter); broadcast and the grouped input/output
 * configurations carry their buffer ids, so the streaming case is a three-argument call
 * and the broadcast / held-operand cases stay a single call:
 *
 *     mul<input(dfb_a), input(dfb_b), output(dfb_out)>(EltwiseShape::tiles(n));
 *     sub<input(dfb_x), input(dfb_row, InputLifecycle::HeldStream), output(dfb_out),
 *         BroadcastDim::Col>(shape);
 *     unary<Exp<>, input(dfb_in), output(dfb_out)>(EltwiseShape::tiles(n));
 *     binary_sfpu<DivBinary<>, input(dfb_a), input(dfb_b), output(dfb_out)>(EltwiseShape::tiles(n));
 *     copy<input(dfb_in), output(dfb_out)>(EltwiseShape::single());
 *
 * The shape argument is an `EltwiseShape`. A bare number is not accepted (the `uint32_t`
 * ctor is `explicit`): write `op<...>(EltwiseShape::tiles(n))`, `EltwiseShape::single()`,
 * or `op<...>(EltwiseShape::grid(Ht, Wt))` so the iteration shape is always explicit.
 *
 * Like `eltwise_chain`, these emit no engine-wide init — the caller owns
 * `compute_kernel_hw_startup(...)` as the first statement of `MAIN()`. Drop to
 * `eltwise_chain` directly for anything outside these shapes (fused multi-op chains,
 * DEST-reuse, fill, etc.).
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// ---------------------------------------------------------------------------
// FPU binary — BinaryFpu(D0) -> PackTile(D0). Op baked into the name.
// Defaults: no broadcast, both operands per-tile streaming.
// ---------------------------------------------------------------------------

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast = BroadcastDim::None>
ALWI void add(EltwiseShape shape);

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast = BroadcastDim::None>
ALWI void sub(EltwiseShape shape);

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast = BroadcastDim::None>
ALWI void mul(EltwiseShape shape);

// ---------------------------------------------------------------------------
// FPU square — x * x, via BinaryFpu reading the one input buffer for both operands
// (the chain's same-buffer path waits/pops it once). Mirrors mul's knobs minus the ones
// that don't apply when both operands are the same tile: no broadcast, and a single
// operand lifecycle / index instead of separate A/B.
// ---------------------------------------------------------------------------

template <InputSpec Input, OutputSpec Output>
ALWI void square(EltwiseShape shape);

// ---------------------------------------------------------------------------
// SFPU unary — CopyTile(D0) -> SfpuOp -> PackTile(D0). SfpuOp is the (DEST-only) op type.
// ---------------------------------------------------------------------------

template <class SfpuOp, InputSpec Input, OutputSpec Output>
ALWI void unary(EltwiseShape shape);

// ---------------------------------------------------------------------------
// SFPU binary — two CopyTile loads (D0, D1) -> SfpuBinOp -> PackTile(D0).
// SfpuBinOp is a DEST-only SFPU binary op type (e.g. DivBinary<>, BinaryMax<>).
// ---------------------------------------------------------------------------

template <class SfpuBinOp, InputSpec AInput, InputSpec BInput, OutputSpec Output>
ALWI void binary_sfpu(EltwiseShape shape);

// ---------------------------------------------------------------------------
// Pure copy — CopyTile(D0) -> PackTile(D0).
// ---------------------------------------------------------------------------

template <InputSpec Input, OutputSpec Output>
ALWI void copy(EltwiseShape shape);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.inl"
