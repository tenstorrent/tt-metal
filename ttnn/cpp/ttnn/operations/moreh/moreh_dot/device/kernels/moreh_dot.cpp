// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        // elemwise-mul
        ACQ();
        cb_wait_front(tt::CBIndex::c_0, onetile);
        cb_wait_front(tt::CBIndex::c_1, onetile);

        cb_reserve_back(tt::CBIndex::c_24, onetile);
        mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        // dst0 = c_in0 x c_in1
        mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        // c_intermed0 = pack(dst0)
        pack_tile(0, tt::CBIndex::c_24);
        cb_push_back(tt::CBIndex::c_24, onetile);

        cb_pop_front(tt::CBIndex::c_0, onetile);
        cb_pop_front(tt::CBIndex::c_1, onetile);
        REL();

        // reduce-w
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            tt::CBIndex::c_24,
            tt::CBIndex::c_2,
            last_out ? tt::CBIndex::c_16 : tt::CBIndex::c_25,
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(tt::CBIndex::c_25, block));
    }
}
