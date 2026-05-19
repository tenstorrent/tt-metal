// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SFPU binary composition + per-side index regime test.
//
// SIDE define selects which CB streams chunk-local vs which is staged upfront.
//   SIDE_A_LOCAL: A=WaitAndPopPerBlock+BlockIter, B=WaitUpfrontPopAtEnd+BlockIter
//   SIDE_B_LOCAL: A=WaitUpfrontPopAtEnd+BlockIter, B=WaitAndPopPerBlock+BlockIter
//
// Because the chain composes CopyTile<D0> + CopyTile<D1> + DEST-only SFPU
// binary, per-side resolution at the CopyTile layer drives correctness — the
// SFPU op is DEST-DEST so it doesn't see CB indexing.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

#define SIDE_A_LOCAL 0
#define SIDE_B_LOCAL 1

#ifndef SIDE
#define SIDE SIDE_A_LOCAL
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

#if SIDE == SIDE_A_LOCAL
    using ATile =
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
    using BTile =
        CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
#else
    using ATile =
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
    using BTile =
        CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
#endif

    eltwise_chain<BLOCK_SIZE>(
        num_tiles,
        ATile{},
        BTile{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerBlockReserveAndPush,
            PackTileIndexMode::BlockIter,
            PackTileReconfig::None>{});
}
