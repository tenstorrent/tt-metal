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
 *     mul<dfb_a, dfb_b, dfb_out>(n);                              // streaming a * b
 *     sub<dfb_x, dfb_max, dfb_out, BroadcastDim::Col,             // softmax x - max
 *         BinaryDataFormatReconfig::Input, OperandKind::Scalar,
 *         InputLifecycle::Streaming, InputLifecycle::HeldStream>(shape);
 *     unary<Exp<>, dfb_in, dfb_out>(n);                           // exp(x)
 *     binary_sfpu<DivBinary<>, dfb_a, dfb_b, dfb_out>(n);         // a / b (SFPU)
 *     copy<dfb_in, dfb_out>(n);
 *
 * The shape argument is an `EltwiseShape` (implicitly built from a plain `uint32_t` tile
 * count), so both `op<...>(n_tiles)` and `op<...>(EltwiseShape::grid(Ht, Wt))` work.
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
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind AIdx = OperandKind::Scalar,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OperandKind BIdx = AIdx,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void add(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Add, Bcast, Reconfig, ALife, BLife, AIdx, Dst::D0, BIdx>{},
        PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind AIdx = OperandKind::Scalar,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OperandKind BIdx = AIdx,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void sub(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Sub, Bcast, Reconfig, ALife, BLife, AIdx, Dst::D0, BIdx>{},
        PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind AIdx = OperandKind::Scalar,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OperandKind BIdx = AIdx,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void mul(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, BinaryFpuOp::Mul, Bcast, Reconfig, ALife, BLife, AIdx, Dst::D0, BIdx>{},
        PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// SFPU unary — CopyTile(D0) -> SfpuOp -> PackTile(D0). SfpuOp is the (DEST-only) op type.
// ---------------------------------------------------------------------------

template <
    class SfpuOp,
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void unary(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(
        shape,
        CopyTile<CbIn, Dst::D0, Life, Idx, Reconfig>{},
        SfpuOp{},
        PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
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
    OperandKind AIdx = OperandKind::Scalar,
    InputLifecycle ALife = InputLifecycle::Streaming,
    InputLifecycle BLife = InputLifecycle::Streaming,
    OperandKind BIdx = AIdx,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void binary_sfpu(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        shape,
        CopyTile<CbA, Dst::D0, ALife, AIdx>{},
        CopyTile<CbB, Dst::D1, BLife, BIdx>{},
        SfpuBinOp{},
        PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// Pure copy — CopyTile(D0) -> PackTile(D0).
// ---------------------------------------------------------------------------

template <
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void copy(EltwiseShape shape) {
    eltwise_chain(
        shape, CopyTile<CbIn, Dst::D0, Life, Idx, Reconfig>{}, PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

// ---------------------------------------------------------------------------
// Unary broadcast — UnaryBcast(D0) -> PackTile(D0). Row/Col/Scalar broadcast of one input.
// ---------------------------------------------------------------------------

template <
    BroadcastDim Dim,
    uint32_t CbIn,
    uint32_t CbOut,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::Input,
    InputLifecycle Life = InputLifecycle::Streaming,
    OutputLifecycle OutLife = OutputLifecycle::Streaming,
    PackTileReconfig OutReconfig = PackTileReconfig::Output>
ALWI void unary_bcast(EltwiseShape shape) {
    eltwise_chain(
        shape, UnaryBcast<Dim, CbIn, Dst::D0, Life, Reconfig>{}, PackTile<CbOut, Dst::D0, OutLife, OutReconfig>{});
}

}  // namespace compute_kernel_lib
