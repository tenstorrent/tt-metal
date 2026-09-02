// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg(args::has_input_grad);
    uint32_t has_other_grad = get_arg(args::has_other_grad);
    uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out0(dfb::out0);
    DataflowBuffer dfb_out1(dfb::out1);

    compute_kernel_hw_startup(dfb::in2, dfb::in0, dfb::out0);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(dfb::in2, dfb::in0);
    dfb_in0.wait_front(onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            dfb_in2.wait_front(onetile);

            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in2, dfb::in0, 0, 0, 0);
            tile_regs_commit();

            dfb_in2.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, dfb::out0);
            tile_regs_release();

            dfb_out0.push_back(onetile);
        }

        if (has_other_grad) {
            dfb_in1.wait_front(onetile);

            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in1, dfb::in0, 0, 0, 0);
            tile_regs_commit();

            dfb_in1.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, dfb::out1);
            tile_regs_release();

            dfb_out1.push_back(onetile);
        }
    }
}
