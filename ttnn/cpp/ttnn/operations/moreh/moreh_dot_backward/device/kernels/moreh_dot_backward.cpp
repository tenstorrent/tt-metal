// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    DataflowBuffer dfb_c0(tt::CBIndex::c_0);
    DataflowBuffer dfb_c1(tt::CBIndex::c_1);
    DataflowBuffer dfb_c2(tt::CBIndex::c_2);
    DataflowBuffer dfb_c16(tt::CBIndex::c_16);
    DataflowBuffer dfb_c17(tt::CBIndex::c_17);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, tt::CBIndex::c_16);
    dfb_c0.wait_front(onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            dfb_c2.wait_front(onetile);

            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, 0, 0, 0);
            tile_regs_commit();

            dfb_c2.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);
            tile_regs_release();

            dfb_c16.push_back(onetile);
        }

        if (has_other_grad) {
            dfb_c1.wait_front(onetile);

            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_1, tt::CBIndex::c_0, 0, 0, 0);
            tile_regs_commit();

            dfb_c1.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_17);
            tile_regs_release();

            dfb_c17.push_back(onetile);
        }
    }
}
