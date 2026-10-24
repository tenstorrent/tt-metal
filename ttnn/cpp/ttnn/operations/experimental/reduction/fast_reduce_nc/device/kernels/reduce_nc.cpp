// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    constexpr auto num_output_tiles = get_arg(args::num_output_tiles);
    constexpr auto num_input_tiles = get_arg(args::num_input_tiles);
    constexpr auto input_granularity = get_arg(args::input_granularity);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    DataflowBuffer cb_in0_obj(dfb::in0);
    DataflowBuffer cb_in1_obj(dfb::in1);
    DataflowBuffer cb_out0_obj(dfb::out0);

    constexpr uint32_t num_input_tiles_iter = num_input_tiles / input_granularity;

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);
    cb_in1_obj.wait_front(onetile);

    // For each assigned output tile, process the input tiles in a doubly nested
    // loop. The inner loop processes the number of tiles specified by
    // input_granularity. The outer loop executes num_input_tiles / input_granularity
    // times.
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        add_init(dfb::in0, dfb::in1, true);
        reconfig_data_format(dfb::in0, dfb::in1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_in0_obj.wait_front(input_granularity);
            for (uint32_t k = 0; k < input_granularity; k++) {
                add_tiles(dfb::in0, dfb::in1, k, first_tile, dst0);
            }
            cb_in0_obj.pop_front(input_granularity);
        }
        tile_regs_commit();
        cb_out0_obj.reserve_back(onetile);
        pack_reconfig_data_format(dfb::out0);
        tile_regs_wait();
        pack_tile(dst0, dfb::out0);
        tile_regs_release();
        cb_out0_obj.push_back(onetile);
    }
    // cb_in1 holds a single broadcast tile waited once and reused across all output tiles;
    // pop it at the end so the CB is left balanced.
    cb_in1_obj.pop_front(onetile);
}
