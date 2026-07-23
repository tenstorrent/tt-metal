// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    DataflowBuffer dfb_in1_obj(cb_in1);  // zero tile
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_in1, cb_in0, cb_out0);
    dfb_in1_obj.wait_front(onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        dfb_in0_obj.wait_front(onetile);
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init_short(cb_in1, cb_in0);
            add_tiles_bcast_scalar(cb_in1, cb_in0, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init_short(cb_in1, cb_in0);
            add_tiles_bcast_rows(cb_in1, cb_in0, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init_short(cb_in1, cb_in0);
            add_tiles_bcast_cols(cb_in1, cb_in0, 0, 0, dst0);
        } else {
            copy_init(cb_in0);
            copy_tile(cb_in0, 0, dst0);
        }
        tile_regs_commit();
        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_out0);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
        dfb_in0_obj.pop_front(onetile);
    }
    dfb_in1_obj.pop_front(onetile);
}
