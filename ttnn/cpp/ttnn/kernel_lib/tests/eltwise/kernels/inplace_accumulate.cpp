// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// In-place CB recurrence test — cb_acc is BOTH the B input AND the pack target.
// Lifecycle: wait_front(cb_acc, 1); compute(cb_in[0], cb_acc[0]) → DEST;
//            cb_reserve_back(cb_acc, 1); pack(dst, cb_acc); cb_pop_front(cb_acc); cb_push_back(cb_acc).
// Chain handles this naturally — BinaryFpu with CbB=cb_acc + PackTile<cb_acc> is the
// "in-place" pattern. CB needs ≥ 2 pages to support wait+reserve simultaneously.

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in  = tt::CBIndex::c_0;
    constexpr uint32_t cb_acc = tt::CBIndex::c_16;  // same CB as input AND pack target

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim   = get_compile_time_arg_val(1);
    const uint32_t num_iters            = per_core_block_count * per_core_block_dim;

    // First iter: seed cb_acc from cb_in (stage 1 — pure copy).
    using SeedChain = EltwiseChain<
        CopyTile<cb_in,  Dst::D0, CopyTilePolicy::WaitAndPop>,
        PackTile<cb_acc, Dst::D0, PackTilePolicy::PerTileReserveAndPush>
    >;
    eltwise_pipeline_init<SeedChain>();
    eltwise_chain(
        1,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        PackTile<cb_acc, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );

    // Subsequent iters: cb_acc <- cb_acc + cb_in (in-place accumulate).
    using AccumElt = BinaryFpu<cb_in, cb_acc, BinaryFpuOp::Add, BroadcastDim::None,
                               BinaryFpuOutputPolicy::PerTile, BinaryDataFormatReconfig::None,
                               CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
                               CbIndexMode::FirstTile, CbIndexMode::FirstTile,
                               Dst::D0, 0, 0, 0, cb_acc>;
    using AccumChain = EltwiseChain<AccumElt,
                                    PackTile<cb_acc, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;

    if (num_iters > 1) {
        eltwise_chain(
            num_iters - 1,
            AccumElt{},
            PackTile<cb_acc, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
        );
    }
}
