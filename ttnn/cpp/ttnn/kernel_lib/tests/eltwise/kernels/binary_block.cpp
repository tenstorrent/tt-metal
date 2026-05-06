// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block-mode binary FPU validation — N tiles processed per acquire/release window.

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

#ifndef BINARY_OP_NAME
#define BINARY_OP_NAME Add
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t block_size = BLOCK_SIZE;
    constexpr BinaryFpuOp op = BinaryFpuOp::BINARY_OP_NAME;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim   = get_compile_time_arg_val(1);
    const uint32_t num_tiles            = per_core_block_count * per_core_block_dim;
    const uint32_t num_blocks           = num_tiles / block_size;

    using BinElt = BlockBinaryFpu<cb_a, cb_b, op, block_size>;
    using PackElt = BlockPackTile<cb_out, block_size>;

    using Chain = EltwiseChain<BinElt, PackElt>;
    eltwise_pipeline_init<Chain>();

    eltwise_chain(num_blocks, BinElt{}, PackElt{});
}
