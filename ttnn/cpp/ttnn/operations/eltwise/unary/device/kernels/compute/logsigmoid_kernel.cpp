// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/logsigmoid.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    DataflowBuffer dfb_in(cb_input);
    DataflowBuffer dfb_out(cb_output);

    compute_kernel_hw_startup(cb_input, cb_output);
    copy_init(cb_input);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();

        copy_init(cb_input);
        copy_tile(cb_input, 0, 0);
        copy_tile(cb_input, 0, 1);

        negative_tile_init();
        negative_tile(1);

        exp_tile_init<true>();
        exp_tile<true>(1);

        logsigmoid_tile_init();
        logsigmoid_tile(0, 1, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
