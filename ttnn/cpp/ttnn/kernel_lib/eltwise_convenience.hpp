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
 * knobs callers actually toggle (`BroadcastDim`, `BinaryDataFormatReconfig`, `CbIndexMode`);
 * other policies use the struct defaults. Drop to `eltwise_chain` for anything outside
 * this surface.
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
    CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void binary_op(uint32_t n_tiles) {
    eltwise_chain(
        n_tiles,
        BinaryFpu<CbA, CbB, Op, Bcast, Reconfig, CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop, Idx>{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void binary_add(uint32_t n) {
    binary_op<BinaryFpuOp::Add, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(n);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void binary_sub(uint32_t n) {
    binary_op<BinaryFpuOp::Sub, CbA, CbB, CbOut, Bcast, Reconfig, Idx>(n);
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    CbIndexMode Idx = CbIndexMode::FirstTile>
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
    CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void unary(uint32_t n_tiles) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp,...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitAndPop, Idx, Reconfig>{},
        SfpuOp{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

// ---- SFPU binary streaming (two CB inputs, DEST-DEST SFPU op, one CB output) ----
// SfpuBinOp must be a DEST-only SFPU binary element (BinaryOp CRTP child),
// e.g. DivBinary, BinaryMax, BinaryMin.
template <class SfpuBinOp, uint32_t CbA, uint32_t CbB, uint32_t CbOut, CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void binary_sfpu(uint32_t n_tiles) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op,...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        n_tiles,
        CopyTile<CbA, Dst::D0, CopyTilePolicy::WaitAndPop, Idx>{},
        CopyTile<CbB, Dst::D1, CopyTilePolicy::WaitAndPop, Idx>{},
        SfpuBinOp{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

// ---- Pure copy ----
template <
    uint32_t CbIn,
    uint32_t CbOut,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    CbIndexMode Idx = CbIndexMode::FirstTile>
ALWI void copy(uint32_t n_tiles) {
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitAndPop, Idx, Reconfig>{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

}  // namespace compute_kernel_lib
