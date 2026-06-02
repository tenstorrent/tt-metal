// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_convenience.hpp
 * @brief Thin convenience entry points — pure inline forwarders to `eltwise_chain`.
 *
 * These wrap the dominant per-tile streaming buckets in one-liner APIs. They are pure
 * chain bodies — caller is responsible for `compute_kernel_hw_startup(...)` as the
 * first statement of `MAIN()` per the D8 caller-init contract. Wrappers expose only the
 * knobs callers actually toggle (`BroadcastDim`, `BinaryDataFormatReconfig`, `OperandKind`);
 * other policies use the struct defaults. Drop to `eltwise_chain` for anything outside
 * this surface.
 *
 * Internal usage of low-level lifecycle constants (`InputLifecycle::Streaming`, `OutputLifecycle::Streaming`, …)
 * matches the public chain element API — see `eltwise_chain.hpp` and
 * `policy_alias_collapse_proposal.md` for the rationale.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace compute_kernel_lib {

// ---- FPU binary streaming (per-tile WaitAndPop on both inputs) ----

template <
    BinaryFpuOp Op,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_op(uint32_t n_tiles) {
    eltwise_chain(
        n_tiles,
        BinaryFpu<CbA, CbB, Op, Bcast, Reconfig, InputLifecycle::Streaming, InputLifecycle::Streaming, Idx>{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_add(uint32_t n) {
    binary_op<BinaryFpuOp::Add, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(n);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_sub(uint32_t n) {
    binary_op<BinaryFpuOp::Sub, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(n);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_mul(uint32_t n) {
    binary_op<BinaryFpuOp::Mul, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(n);
}

// ---- SFPU unary streaming ----
// SfpuOp must be a DEST-only SFPU element (UnaryOp CRTP child).
template <
    class SfpuOp,
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void unary(uint32_t n_tiles) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp,...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, InputLifecycle::Streaming, Idx, Reconfig>{},
        SfpuOp{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

// ---- SFPU binary streaming (two CB inputs, DEST-DEST SFPU op, one CB output) ----
// SfpuBinOp must be a DEST-only SFPU binary element (BinaryOp CRTP child),
// e.g. DivBinary, BinaryMax, BinaryMin.
template <class SfpuBinOp, uint32_t CbA, uint32_t CbB, uint32_t CbOut, OperandKind Idx = OperandKind::Scalar>
ALWI void binary_sfpu(uint32_t n_tiles) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op,...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        n_tiles,
        CopyTile<CbA, Dst::D0, InputLifecycle::Streaming, Idx>{},
        CopyTile<CbB, Dst::D1, InputLifecycle::Streaming, Idx>{},
        SfpuBinOp{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

// ---- Pure copy ----
template <
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void copy(uint32_t n_tiles) {
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, InputLifecycle::Streaming, Idx, Reconfig>{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

// =============================================================================
// 2D shape overloads — pass `EltwiseShape::of(Ht, Wt)` for row/col/scalar
// broadcast indexing. Forwarders to the 2D `eltwise_chain(EltwiseShape, …)`
// overload; same default policies as the 1D entries above.
// =============================================================================

template <
    BinaryFpuOp Op,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_op(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<CbA, CbB, Op, Bcast, Reconfig, InputLifecycle::Streaming, InputLifecycle::Streaming, Idx>{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_add(EltwiseShape shape) {
    binary_op<BinaryFpuOp::Add, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(shape);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_sub(EltwiseShape shape) {
    binary_op<BinaryFpuOp::Sub, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(shape);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void binary_mul(EltwiseShape shape) {
    binary_op<BinaryFpuOp::Mul, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(shape);
}

template <
    class SfpuOp,
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void unary(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp,...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(
        shape,
        CopyTile<CbIn, Dst::D0, InputLifecycle::Streaming, Idx, Reconfig>{},
        SfpuOp{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

template <class SfpuBinOp, uint32_t CbA, uint32_t CbB, uint32_t CbOut, OperandKind Idx = OperandKind::Scalar>
ALWI void binary_sfpu(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op,...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        shape,
        CopyTile<CbA, Dst::D0, InputLifecycle::Streaming, Idx>{},
        CopyTile<CbB, Dst::D1, InputLifecycle::Streaming, Idx>{},
        SfpuBinOp{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

template <
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind Idx = OperandKind::Scalar>
ALWI void copy(EltwiseShape shape) {
    eltwise_chain(
        shape,
        CopyTile<CbIn, Dst::D0, InputLifecycle::Streaming, Idx, Reconfig>{},
        PackTile<CbOut, OutputLifecycle::Streaming>{});
}

}  // namespace compute_kernel_lib
