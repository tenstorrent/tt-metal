// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    DataflowBuffer dfb_c0(tt::CBIndex::c_0);
    DataflowBuffer dfb_c1(tt::CBIndex::c_1);
    DataflowBuffer dfb_c2(tt::CBIndex::c_2);
    DataflowBuffer dfb_c24(tt::CBIndex::c_24);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        dfb_c0.wait_front(onetile);
        dfb_c1.wait_front(onetile);

        tile_regs_acquire();
        mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        tile_regs_commit();

        dfb_c0.pop_front(onetile);
        dfb_c1.pop_front(onetile);

        dfb_c24.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_24);
        tile_regs_release();

        dfb_c24.push_back(onetile);

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
    // The reduce helper waits on the scaler CB (c_2) each block but never pops it; the single
    // scaler tile is reused across all blocks. Pop it once at the end to balance the CB.
    dfb_c2.pop_front(onetile);
}
