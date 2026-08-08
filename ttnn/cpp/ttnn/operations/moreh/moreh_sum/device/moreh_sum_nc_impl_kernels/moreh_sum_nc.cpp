// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    DataflowBuffer dfb_in1_obj(cb_in1);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t idx0 = 0;
    constexpr bool acc_to_dest = true;

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    dfb_in1_obj.wait_front(onetile);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        add_init(cb_in0, cb_in1, acc_to_dest);
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            dfb_in0_obj.wait_front(onetile);
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_in0, cb_in1);
#endif
            add_tiles(cb_in0, cb_in1, idx0, idx0, dst0);
            dfb_in0_obj.pop_front(onetile);
        }
        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
#if defined FP32_DEST_ACC_EN
        pack_reconfig_data_format(cb_out0);
#endif
        pack_tile(dst0, cb_out0);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
    }
}
