// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BlockSize > 1 validation:
//   eltwise_chain<DEST_AUTO_LIMIT>(N, CopyTile<…, WaitUpfrontPopAtEnd, BlockIter, CbIndexMode::FirstTile,
//   CopyTileReconfig::None>{},
//                                Exp{}, PackTile<…, UpfrontReservePushAtEnd, BlockIter, PackTileIndexMode::FirstTile,
//                                PackTileReconfig::None>{})
//
// chain_lane_width = 1 (all elements at Dst::D0). BlockSize = DEST_AUTO_LIMIT / 1.
// Reader pushes N tiles upfront; chain processes BlockSize tiles per outer iter via
// DEST lanes (lane j → DEST[j]); pack writes N tiles, pushes at end.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);

    eltwise_chain<DEST_AUTO_LIMIT>(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter, CopyTileReconfig::None>{},
        Exp<>{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::UpfrontReservePushAtEnd,
            PackTileIndexMode::BlockIter,
            PackTileReconfig::None>{});
}
