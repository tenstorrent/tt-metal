// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool is_last = (block == per_core_block_cnt - 1);

        // elemwise-mul
        compute_kernel_lib::mul(
            tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24, compute_kernel_lib::BinaryTileShape::single());

        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputMode::STREAMING,
            compute_kernel_lib::ReduceDataFormatReconfig::NONE>(
            tt::CBIndex::c_24,
            tt::CBIndex::c_2,
            is_last ? tt::CBIndex::c_16 : tt::CBIndex::c_25,
            compute_kernel_lib::TileShape::single(),
            {},
            compute_kernel_lib::Accumulate::at(tt::CBIndex::c_25, block));
    }
}
