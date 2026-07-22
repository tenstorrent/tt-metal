// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/sfpu_int_sum.h"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    DataflowBuffer dfb_intermed0_obj(cb_intermed0);
    constexpr int onetile = 1;
    constexpr int idx0 = 0;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    compute_kernel_hw_startup(cb_in0, cb_out0);
    copy_init(cb_in0);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);
            tile_regs_acquire();
            copy_tile_to_dst(dfb_in0_obj, idx0, dst0);
            if (enable_reload) {
                copy_tile_to_dst(dfb_intermed0_obj, idx0, dst1);
                sfpu_sum_int_init();
                sfpu_add_int(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t cb_out = (last_out) ? (cb_out0) : (cb_intermed0);
            pack_tile_from_dst(DataflowBuffer(cb_out), dst0);
            tile_regs_release();
            enable_reload = true;
        }
    }
}
