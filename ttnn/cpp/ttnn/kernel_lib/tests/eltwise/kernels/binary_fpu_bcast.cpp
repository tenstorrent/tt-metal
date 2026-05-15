// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BinaryFpu broadcast validation — selects op via BINARY_OP_NAME and dim via BCAST_DIM.

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef BINARY_OP_NAME
#define BINARY_OP_NAME Add
#endif

#ifndef BCAST_DIM
#define BCAST_DIM Row
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr BinaryFpuOp op = BinaryFpuOp::BINARY_OP_NAME;
    constexpr BroadcastDim dim = BroadcastDim::BCAST_DIM;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        op,
        dim,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitNoPop,
        CbIndexMode::FirstTile,
        Dst::D0>;

    eltwise_chain(
        num_tiles,
        BinElt{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
