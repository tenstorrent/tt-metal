// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SFPU binary chain composition smoke test (proposal Step 0).
//
// Validates that the existing chain helper composes:
//   CopyTile<cb_a, …, Dst::D0>     → A into even DEST slots
//   CopyTile<cb_b, …, Dst::D1>     → B into odd DEST slots; srca fold reconfigs cb_a→cb_b
//   AddBinary<Dst::D0, Dst::D1, Dst::D0>  → existing DEST-only SFPU binary
//   PackTile<cb_out, Dst::D0, …>   → pack even slots
//
// at `chain_lane_width = 2`. Result is `cb_a + cb_b` per tile.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    eltwise_chain<BLOCK_SIZE>(
        num_tiles,
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>{},
        CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerBlockReserveAndPush,
            PackTileIndexMode::BlockIter,
            PackTileReconfig::None>{});
}
