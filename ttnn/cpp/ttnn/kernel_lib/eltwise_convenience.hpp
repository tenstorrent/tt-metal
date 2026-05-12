// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_convenience.hpp
 * @brief Thin convenience entry points — pure inline forwarders to `eltwise_chain`.
 *
 * These wrap the dominant TSV buckets in one-liner APIs. They have no policy enums of their own;
 * any non-default behavior should drop to `eltwise_chain` directly.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// ---- FPU binary streaming (per-tile WaitAndPop on both inputs) ----

template <BinaryFpuOp Op, uint32_t CbA, uint32_t CbB, uint32_t CbOut>
ALWI void binary_op(uint32_t n_tiles) {
    using BinElt = BinaryFpu<
        CbA,
        CbB,
        CbOut,
        Op,
        BroadcastDim::None,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        Dst::D0>;
    // D8: caller-side BIG init. The convenience wrapper boots the engine for the
    // (CbA, CbB, CbOut) triple it owns, then runs the chain (per-element-init only).
    compute_kernel_hw_startup(CbA, CbB, CbOut);
    eltwise_chain(n_tiles, BinElt{}, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

template <uint32_t CbA, uint32_t CbB, uint32_t CbOut>
ALWI void binary_add(uint32_t n_tiles) {
    binary_op<BinaryFpuOp::Add, CbA, CbB, CbOut>(n_tiles);
}

template <uint32_t CbA, uint32_t CbB, uint32_t CbOut>
ALWI void binary_sub(uint32_t n_tiles) {
    binary_op<BinaryFpuOp::Sub, CbA, CbB, CbOut>(n_tiles);
}

template <uint32_t CbA, uint32_t CbB, uint32_t CbOut>
ALWI void binary_mul(uint32_t n_tiles) {
    binary_op<BinaryFpuOp::Mul, CbA, CbB, CbOut>(n_tiles);
}

// ---- Unary SFPU streaming ----
template <class SfpuOp, uint32_t CbIn, uint32_t CbOut>
ALWI void unary_op(uint32_t n_tiles) {
    // D8 caller-side BIG init.
    compute_kernel_hw_startup(CbIn, CbIn, CbOut);
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        SfpuOp{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

// ---- Pure copy ----
template <uint32_t CbIn, uint32_t CbOut>
ALWI void copy(uint32_t n_tiles) {
    // D8 caller-side BIG init.
    compute_kernel_hw_startup(CbIn, CbIn, CbOut);
    eltwise_chain(
        n_tiles,
        CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

}  // namespace compute_kernel_lib
