// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    CircularBuffer cb_c0(tt::CBIndex::c_0);
    CircularBuffer cb_c1(tt::CBIndex::c_1);
    CircularBuffer cb_c24(tt::CBIndex::c_24);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        cb_c0.wait_front(onetile);
        cb_c1.wait_front(onetile);

        tile_regs_acquire();
        mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        tile_regs_commit();

        cb_c0.pop_front(onetile);
        cb_c1.pop_front(onetile);

        cb_c24.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_24);
        tile_regs_release();

        cb_c24.push_back(onetile);

        // reduce-w
        if (last_out) {
            compute_kernel_lib::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                tt::CBIndex::c_24,
                tt::CBIndex::c_2,
                tt::CBIndex::c_16,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(tt::CBIndex::c_25, block));
        } else {
            compute_kernel_lib::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                tt::CBIndex::c_24,
                tt::CBIndex::c_2,
                tt::CBIndex::c_25,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(tt::CBIndex::c_25, block));
        }
    }
}
