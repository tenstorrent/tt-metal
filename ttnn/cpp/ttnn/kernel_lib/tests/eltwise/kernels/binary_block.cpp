// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streaming binary FPU + per-tile pack — formerly the Block-mode validation.
// BLOCK_SIZE define preserved for test parameterization but unused in the kernel
// (chain runs per-tile streaming over num_tiles total).

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

#ifndef BINARY_OP_NAME
#define BINARY_OP_NAME Add
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr BinaryFpuOp op = BinaryFpuOp::BINARY_OP_NAME;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using BinElt = BinaryFpu<cb_a, cb_b, op, BroadcastDim::None, BinaryDataFormatReconfig::None>;
    using PackElt = PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>;

    eltwise_chain(num_tiles, BinElt{}, PackElt{});
}
