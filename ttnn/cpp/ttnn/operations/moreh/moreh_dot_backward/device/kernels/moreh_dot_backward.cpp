// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);

    with_nullable_token(dfb::out0, [&](const DFBBindingToken& out0) {
        compute_kernel_hw_startup(dfb::in2, dfb::in0, out0);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(dfb::in2, dfb::in0);
    });
    with_nullable_token(dfb::out1, [&](const DFBBindingToken& out1) {
        if constexpr (is_null_binding(dfb::out0)) {
            compute_kernel_hw_startup(dfb::in1, dfb::in0, out1);
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(dfb::in1, dfb::in0);
        }
    });

    dfb_in0.wait_front(onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in2.wait_front(onetile);
        with_nullable_token(dfb::out0, [&](const DFBBindingToken& out0) {
            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in2, dfb::in0, 0, 0, 0);
            tile_regs_commit();

            dfb_in2.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, out0);
            tile_regs_release();

            DataflowBuffer dfb_out0(out0);
            dfb_out0.push_back(onetile);
        });
        if constexpr (is_null_binding(dfb::out0)) {
            dfb_in2.pop_front(onetile);
        }

        dfb_in1.wait_front(onetile);
        with_nullable_token(dfb::out1, [&](const DFBBindingToken& out1) {
            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in1, dfb::in0, 0, 0, 0);
            tile_regs_commit();

            dfb_in1.pop_front(onetile);

            tile_regs_wait();
            pack_tile(0, out1);
            tile_regs_release();

            DataflowBuffer dfb_out1(out1);
            dfb_out1.push_back(onetile);
        });
        if constexpr (is_null_binding(dfb::out1)) {
            dfb_in1.pop_front(onetile);
        }
    }
}
