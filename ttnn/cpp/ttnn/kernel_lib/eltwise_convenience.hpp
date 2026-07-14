// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_convenience.hpp
 * @brief One-liner entry points for the dominant eltwise chain shapes.
 *
 * Each wrapper is a pure inline forwarder to `eltwise_chain` for one common shape, so a
 * simple op needs one call instead of a hand-written chain. The op is baked into the name
 * (`add`/`sub`/`mul`, or the SFPU op as a type parameter); the per-operand lifecycle,
 * broadcast and operand-kind are defaulted template parameters, so the streaming case is a
 * three-argument call and the broadcast / held-operand cases stay a single call:
 *
 *     mul<dfb_a, dfb_b, dfb_out>(EltwiseShape::tiles(n));         // streaming a * b
 *     sub<dfb_x, dfb_max, dfb_out, BroadcastDim::Col,             // softmax x - max
 *         InputLifecycle::Streaming, InputLifecycle::HeldStream>(shape);
 *     unary<Exp<>, dfb_in, dfb_out>(EltwiseShape::tiles(n));      // exp(x)
 *     binary_sfpu<DivBinary<>, dfb_a, dfb_b, dfb_out>(EltwiseShape::tiles(n)); // a / b (SFPU)
 *     copy<dfb_in, dfb_out>(EltwiseShape::single());             // one tile
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace compute_kernel_lib {

// ---------------------------------------------------------------------------
// FPU binary — BinaryFpu(D0) -> PackTile(D0). Op baked into the name.
// Defaults: no broadcast, both operands per-tile streaming.
// ---------------------------------------------------------------------------

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind AIdx = OperandKind::Scalar,
    OperandKind BIdx = AIdx>
ALWI void add(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Add, Bcast, ALife, BLife, Reconfig, Dst::D0, AIdx, BIdx>{},
        PackTile<CbOut, OutLife, OutReconfig>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind AIdx = OperandKind::Scalar,
    OperandKind BIdx = AIdx>
ALWI void sub(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Sub, Bcast, ALife, BLife, Reconfig, Dst::D0, AIdx, BIdx>{},
        PackTile<CbOut, OutLife, OutReconfig>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind AIdx = OperandKind::Scalar,
    OperandKind BIdx = AIdx>
ALWI void mul(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Mul, Bcast, ALife, BLife, Reconfig, Dst::D0, AIdx, BIdx>{},
        PackTile<CbOut, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// FPU square — x * x, via BinaryFpu reading the one input buffer for both operands
// (the chain's same-buffer path waits/pops it once). Mirrors mul's knobs minus the ones
// that don't apply when both operands are the same tile: no broadcast, and a single
// operand lifecycle / index instead of separate A/B.
// ---------------------------------------------------------------------------

template <
    uint32_t CbIn,
    uint32_t CbOut,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind Idx = OperandKind::Scalar>
ALWI void square(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbIn, CbIn, BinaryFpuOp::Mul, BroadcastDim::None, Life, Life, Reconfig, Dst::D0, Idx, Idx>{},
        PackTile<CbOut, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// SFPU unary — CopyTile(D0) -> SfpuOp -> PackTile(D0). SfpuOp is the (DEST-only) op type.
// ---------------------------------------------------------------------------

template <
    class SfpuOp,
    uint32_t CbIn,
    uint32_t CbOut,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind Idx = OperandKind::Scalar>
ALWI void unary(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(
        shape, CopyTile<CbIn, Dst::D0, Life, Reconfig, Idx>{}, SfpuOp{}, PackTile<CbOut, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// SFPU binary — two CopyTile loads (D0, D1) -> SfpuBinOp -> PackTile(D0).
// SfpuBinOp is a DEST-only SFPU binary op type (e.g. DivBinary<>, BinaryMax<>).
// ---------------------------------------------------------------------------

template <
    class SfpuBinOp,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind AIdx = OperandKind::Scalar,
    OperandKind BIdx = AIdx>
ALWI void binary_sfpu(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        shape,
        CopyTile<CbA, Dst::D0, ALife, CopyTileReconfig::Input, AIdx>{},
        CopyTile<CbB, Dst::D1, BLife, CopyTileReconfig::Input, BIdx>{},
        SfpuBinOp{},
        PackTile<CbOut, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// Pure copy — CopyTile(D0) -> PackTile(D0).
// ---------------------------------------------------------------------------

template <
    uint32_t CbIn,
    uint32_t CbOut,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output,
    OperandKind Idx = OperandKind::Scalar>
ALWI void copy(EltwiseShape shape) {
    eltwise_chain(shape, CopyTile<CbIn, Dst::D0, Life, Reconfig, Idx>{}, PackTile<CbOut, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// Unary broadcast — UnaryBcast(D0) -> PackTile(D0). Row/Col/Scalar broadcast of one input.
// ---------------------------------------------------------------------------

template <
    BroadcastDim Dim,
    uint32_t CbIn,
    uint32_t CbOut,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::Input,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void unary_bcast(EltwiseShape shape) {
    eltwise_chain(shape, UnaryBcast<Dim, CbIn, Life, Reconfig>{}, PackTile<CbOut, OutLife, OutReconfig>{});
}

}  // namespace compute_kernel_lib
