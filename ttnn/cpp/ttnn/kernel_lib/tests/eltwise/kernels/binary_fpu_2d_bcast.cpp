// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// 2D `eltwise_chain` validation kernel — exercises (Ht, Wt) tile-grid walk with
// per-element broadcast indexing (ColBcast / RowBcast / Scalar B).
//
// Compile-time defines:
//   TEST_HT, TEST_WT       — tile grid dimensions
//   TEST_B_INDEX_MODE      — CbIndexMode value for B side (4=RowBcast, 5=ColBcast, 0=FirstTile/Scalar)

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef TEST_HT
#define TEST_HT 4
#endif
#ifndef TEST_WT
#define TEST_WT 8
#endif
// Default to ColBcast (matches enum order in CbIndexMode).
#ifndef TEST_B_INDEX_MODE
#define TEST_B_INDEX_MODE 5
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = TEST_HT;
    constexpr uint32_t Wt = TEST_WT;
    constexpr CbIndexMode b_idx_mode = static_cast<CbIndexMode>(TEST_B_INDEX_MODE);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    eltwise_chain(
        EltwiseShape::of(Ht, Wt),
        BinaryFpu<
            cb_a,
            cb_b,
            BinaryFpuOp::Sub,
            BroadcastDim::None,
            BinaryDataFormatReconfig::None,
            CopyTilePolicy::WaitUpfrontPopAtEnd,  // A: wait Ht*Wt upfront, pop at end
            CopyTilePolicy::WaitUpfrontPopAtEnd,  // B: wait window upfront, pop at end
            CbIndexMode::BlockIter,               // AIndex — walks ht*Wt + wt
            Dst::D0,
            b_idx_mode>{},  // BIndex — Row/Col/Scalar
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
