// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Same-CB BinaryFpu validation kernel — both operands feed from cb_a (cb_b == cb_a).
// Helper must dedup `cb_wait_front` / `cb_pop_front`.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef BINARY_OP_NAME
#define BINARY_OP_NAME Mul
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr BinaryFpuOp op = BinaryFpuOp::BINARY_OP_NAME;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    using BinElt = BinaryFpu<
        cb_in,
        cb_in,
        op,
        BroadcastDim::None,
        BinaryFpuOutputPolicy::PerTile,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        CbIndexMode::FirstTile,
        Dst::D0>;

    using Chain = EltwiseChain<BinElt, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();

    eltwise_chain(num_tiles, BinElt{}, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
