// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// `BinaryFpu` per-side local-vs-absolute index toggle — 2D variant with
// row-broadcast on B. Models the post-allgather beta-add shape where B is a
// Wt-wide vector replicated across rows.
//
// A = WaitAndPopPerBlock + BlockIter (chunk-local in 2D).
// B = WaitUpfrontPopAtEnd + RowBcast (B has Wt tiles, walked by absolute wt).
// Pack = PerBlockReserveAndPush + BlockIter (chunk-local).
//
// Compile-time defines:
//   TEST_HT, TEST_WT — 2D grid dimensions

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef TEST_HT
#define TEST_HT 4
#endif
#ifndef TEST_WT
#define TEST_WT 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = TEST_HT;
    constexpr uint32_t Wt = TEST_WT;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    // BroadcastDim::None — regular elementwise tile add; CbIndexMode::RowBcast
    // on B picks a different B tile per column (B holds Wt full tiles, each
    // applied to its column in every row).
    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPopPerBlock,   // A: chunk-local
        CopyTilePolicy::WaitUpfrontPopAtEnd,  // B: held upfront
        CbIndexMode::BlockIter,               // AIndex
        Dst::D0,
        CbIndexMode::RowBcast>;  // BIndex — wt (absolute)
    using PackElt = PackTile<
        cb_out,
        Dst::D0,
        PackTilePolicy::PerBlockReserveAndPush,
        PackTileIndexMode::BlockIter,
        PackTileReconfig::None>;

    eltwise_chain<BLOCK_SIZE>(EltwiseShape::of(Ht, Wt), BinElt{}, PackElt{});
}
