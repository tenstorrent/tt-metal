// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    // num_output_tiles carries the per-core work-split count (the host's num_cols_per_core_group_N).
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr uint32_t num_input_tiles = get_arg(args::num_input_tiles);

    DataflowBuffer dfb_in0_obj(dfb::input);
    DataflowBuffer dfb_in1_obj(dfb::zero);
    DataflowBuffer dfb_out0_obj(dfb::out);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t idx0 = 0;
    constexpr bool acc_to_dest = true;

    compute_kernel_hw_startup(dfb::input, dfb::zero, dfb::out);
    dfb_in1_obj.wait_front(onetile);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        add_init(dfb::input, dfb::zero, acc_to_dest);
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            dfb_in0_obj.wait_front(onetile);
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(dfb::input, dfb::zero);
#endif
            add_tiles(dfb::input, dfb::zero, idx0, idx0, dst0);
            dfb_in0_obj.pop_front(onetile);
        }
        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
#if defined FP32_DEST_ACC_EN
        pack_reconfig_data_format(dfb::out);
#endif
        pack_tile(dst0, dfb::out);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
    }
}
